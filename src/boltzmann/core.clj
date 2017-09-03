(ns boltzmann.core
  (:require
    [boltzmann.utils :refer :all]
    [boltzmann.optimizers :refer :all]
    [boltzmann.inference :refer :all]
    [uncomplicate.neanderthal.core :refer [imax scal sum axpy rk entry! mv trans xpy axpby! amax]]
    [uncomplicate.neanderthal.native :refer [dv dge]]
    [clojure.core.matrix.random :refer [sample-normal]]
    [clojure-csv.core :refer [parse-csv]]
    [clojure.pprint :refer [pprint]])
  (:gen-class))

(def hidden-width 30)
(def input-size (* 28 28))
(def num-training-examples 60000)
(def weight-sd 0.01)
(comment (def output-weight-sd 0.3))
(def test-iterations 10)
(def num-epochs 5)

(def thickness [1.0 [3.0] 5.0])

(defn activations-from-csv-row [[label-string & pixel-strings]]
  (scal-rec thickness
            (let [label (Integer. label-string)
                  pixels (eager-map #(/ (Integer. %) 255.0) pixel-strings)]
              [(dv pixels)
               [(dv (repeat hidden-width 0.5))]
               (entry! (dv 10) label 1.0)])))

(def initial-activations-list (->> "mnist_train.csv"
                                          slurp
                                          parse-csv
                                          (take num-training-examples)
                                          (eager-map activations-from-csv-row)))


;;TODO: initial SD dependent on thickness
(def initial-weights [(dge hidden-width input-size
                           (eager-map #(* % weight-sd) (sample-normal (* hidden-width input-size))))
                      []
                      (dge 10 hidden-width
                           (eager-map #(* % weight-sd) (sample-normal (* 10 hidden-width))))])

(def initial-biases [(dv input-size) [(dv hidden-width)] (dv 10)])

(def initial-params [initial-weights initial-biases])

(def initial-optimizer
  (init-opt-A
    (eager-map (partial infer thickness initial-params)
               (take 1000 initial-activations-list))))

(defn training-iteration [thickness [[params
                            optimizer]
                           activations-acc]
                          next-activations]
  (let [[awake-unit-activations _ :as all-activations] (infer thickness params next-activations)]
    (occasionally 1.5 (println count params))
    [(update-params-and-optimizer optimizer params all-activations)
     (conj activations-acc awake-unit-activations)]))

(defn test-activations [pixels]
  (scal-rec thickness
            [(dv pixels)
             [(dv (repeat hidden-width 0.5))]
             (dv (repeat 10 0.1))]))

(defn test-csv-row [params [label-string & pixel-strings]]
  (= (imax (output-probabilities params
                                 ((func-power
                                    (partial test-gibbs thickness params)
                                    test-iterations)
                                   (test-activations
                                     (eager-map #(/ (Integer. %) 255.0) pixel-strings)))))
     (Integer. label-string)))

(defn test-all [params]
  (println "testing...")
  (println "num correct: "
    (count (filter identity (eager-map (partial test-csv-row params) (parse-csv (slurp "mnist_test.csv")))))))

(defn training-epoch [thickness [[initial-params initial-optimizer] activations-list]]
  (println "doing epoch")
  (doto
    (reduce (partial training-iteration thickness)
            [[initial-params initial-optimizer] []]
            activations-list)
    ((comp test-all first first))))

(defn -main
  [& args]
  ((func-power (partial training-epoch thickness) num-epochs)
    [[initial-params initial-optimizer] initial-activations-list]))

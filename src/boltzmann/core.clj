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
(def input-weight-sd 0.03)
(def output-weight-sd 0.3)
(def test-iterations 10)

(defn activations-from-csv-row [[label-string & pixel-strings]]
  (let [label (Integer. label-string)
        pixels (eager-map #(/ (Integer. %) 255.0) pixel-strings)]
    [(dv pixels)
     [(dv (repeat hidden-width 0.5))]
     (entry! (dv 10) label 1.0)]))

(def initial-activations-list (->> "mnist_train.csv"
                                   slurp
                                   parse-csv
                                   (take num-training-examples)
                                   (eager-map activations-from-csv-row)))

(def initial-weights [(dge hidden-width input-size
                           (eager-map #(* % input-weight-sd) (sample-normal (* hidden-width input-size))))
                      []
                      (dge 10 hidden-width
                           (eager-map #(* % output-weight-sd) (sample-normal (* 10 hidden-width))))])

(def initial-biases [(dv input-size) [(dv hidden-width)] (dv 10)])

(def initial-params [initial-weights initial-biases])

(def initial-optimizer
  (init-opt-A
    (eager-map (partial infer initial-params)
               (take 1000 initial-activations-list))))

(defn training-iteration [[params
                           optimizer
                           activations-acc]
                          next-activations]
  (let [[_ [awake-unit-activations _] :as all-activations] (infer params next-activations)
        new-optimizer (update-optimizer optimizer all-activations)]
    (occasionally 1.5 (println count params))
    [(update-params optimizer params all-activations)
     new-optimizer
     (conj activations-acc awake-unit-activations)]))

(defn training-epoch [[initial-params initial-moments activations-list]]
  (println "doing epoch")
  (reduce training-iteration [initial-params initial-moments []] activations-list))

(defn test-activations [pixels]
  [(dv pixels)
   [(dv (repeat hidden-width 0.5))]
   (dv (repeat 10 0.1))])

(defn test-csv-row [params [label-string & pixel-strings]]
  (let [label (Integer. label-string)
        pixels (eager-map #(/ (Integer. %) 255.0) pixel-strings)
        [in-act hidden-act out-act] ((func-power (partial test-gibbs params) test-iterations)
                                      (test-activations pixels))]
    (= (imax out-act) label)))

(defn test-all [params]
  (println "testing...")
  (println (count (filter identity (eager-map (partial test-csv-row params) (parse-csv (slurp "mnist_test.csv")))))))

(defn -main
  [& args]
  (-> [initial-params initial-optimizer initial-activations-list]
      ((func-power training-epoch 1)) first test-all))

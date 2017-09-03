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
(def weight-sd 0.03)
(comment (def output-weight-sd 0.3))
(def test-iterations 10)

(def thickness [1.0 [3.0] 5.0])

(defn scaled-activations-from-csv-row [[label-string & pixel-strings]]
  (scal-rec thickness
            (let [label (Integer. label-string)
                  pixels (eager-map #(/ (Integer. %) 255.0) pixel-strings)]
              [(dv pixels)
               [(dv (repeat hidden-width 0.5))]
               (entry! (dv 10) label 1.0)])))

(def initial-scaled-activations-list (->> "mnist_train.csv"
                                          slurp
                                          parse-csv
                                          (take num-training-examples)
                                          (eager-map scaled-activations-from-csv-row)))

(def initial-weights [(dge hidden-width input-size
                           (eager-map #(* % weight-sd) (sample-normal (* hidden-width input-size))))
                      []
                      (dge 10 hidden-width
                           (eager-map #(* % weight-sd) (sample-normal (* 10 hidden-width))))])

(def initial-biases [(dv input-size) [(dv hidden-width)] (dv 10)])

(def initial-params [[initial-weights initial-biases] thickness])

(def initial-optimizer
  (init-opt-A
    (eager-map (partial infer-and-scale initial-params)
               (take 1000 initial-scaled-activations-list))))

(defn training-iteration [[[params
                            optimizer]
                           scaled-activations-acc]
                          next-scaled-activations]
  (let [[scaled-awake-unit-activations _ :as all-scaled-activations] (infer-and-scale params next-scaled-activations)]
    (occasionally 1.5 (println count params))
    [(update-params-and-optimizer optimizer params all-scaled-activations)
     (conj scaled-activations-acc scaled-awake-unit-activations)]))

(defn training-epoch [[[initial-params initial-optimizer] scaled-activations-list]]
  (println "doing epoch")
  (reduce training-iteration [[initial-params initial-optimizer] []] scaled-activations-list))

(defn scaled-test-activations [pixels]
  (scal-rec thickness
            [(dv pixels)
             [(dv (repeat hidden-width 0.5))]
             (dv (repeat 10 0.1))]))

(defn test-csv-row [params [label-string & pixel-strings]]
  (let [label (Integer. label-string)
        pixels (eager-map #(/ (Integer. %) 255.0) pixel-strings)
        [in-act hidden-act out-act] (test-gibbs params
                                                ((func-power (partial scaled-test-gibbs params) test-iterations)
                                                  (scaled-test-activations pixels)))]
    (= (imax out-act) label)))

(defn test-all [params]
  (println "testing...")
  (println (count (filter identity (eager-map (partial test-csv-row params) (parse-csv (slurp "mnist_test.csv")))))))

(defn -main
  [& args]
  (-> [[initial-params initial-optimizer] initial-scaled-activations-list]
      ((func-power training-epoch 20)) first first test-all))

(ns boltzmann.core
  (:require [uncomplicate.neanderthal.core :refer [imax scal sum axpy rk entry! mv trans xpy]]
            [uncomplicate.neanderthal.native :refer [dv dge]]
            [uncomplicate.neanderthal.real :refer [entry]]
            [uncomplicate.fluokitten.core :refer [fmap]]
            [clojure.core.matrix.random :refer [sample-normal]]
            [clojure-csv.core :refer [parse-csv]]
            [clojure.pprint :refer [pprint]])
  (:gen-class))

(comment
  (def x (dv 1 2 3))
  (def y (dv 10 52 30))
  (def ones (dv 1.0 1.0 1.0)))

(def hidden-width 30)
(def ones (dv (repeat 10 1.0)))
(def input-size (* 28 28))
(def awake-iterations 1)
(def dream-iterations 5)
(def learning-rate 0.001)
(def num-training-examples 60000) ;; actually 60000
(def weight-sd 0.03)
(def test-iterations 10)

(defn activations-from-csv-row [[label-string & pixel-strings]]
  (let [label (Integer. label-string)
        pixels (map #(/ (Integer. %) 255.0) pixel-strings)]
    [(dv pixels)
     [(dv (repeat hidden-width 0.5))]
     (entry! (dv 10) label 1.0)]))

(defn test-activations [pixels]
  [(dv pixels)
   [(dv (repeat hidden-width 0.5))]
   (dv (repeat 10 0.5))])

(def initial-activations (->> "mnist_train.csv"
                              slurp
                              parse-csv
                              (take num-training-examples)
                              (map activations-from-csv-row)
                              vec))

(def initial-weights [(dge hidden-width input-size
                           (map #(* % weight-sd) (sample-normal (* hidden-width input-size))))
                      [] (dge 10 hidden-width)])

(def initial-biases [(dv input-size) [(dv hidden-width)] (dv 10)])

(def initial-params [initial-weights initial-biases])

(defn exp ^double [^double x]
  (Math/exp x))

(defn vexp [v]
  (fmap exp v))

(defn vmax [v]
  (entry v (imax v)))

(defn normalize [v]
  (scal (/ 1.0 (sum v)) v))

(defn softmax [v]
  (normalize (vexp (axpy (- (vmax v)) ones v))))

(defn logistic-sigmoid ^double [^double x]
  (/ 1.0 (+ 1.0 (Math/exp (- x)))))

(defn vsig [v]
  (fmap logistic-sigmoid v))

(defn middle [v]
  (-> v rest drop-last vec))

(defn trans-mv [a v]
  (mv (trans a) v))
;;use first, then middle, then last

(defn in-logits [[[in-weights hidden-weights out-weights] [in-biases hidden-biases out-biases]]
                 [in-act hidden-act out-act]]
  (xpy in-biases (trans-mv in-weights (first hidden-act))))

(defn out-logits [[[in-weights hidden-weights out-weights] [in-biases hidden-biases out-biases]]
                  [in-act hidden-act out-act]]
  (xpy out-biases (mv out-weights (last hidden-act))))

(defn hidden-logits [[[in-weights hidden-weights out-weights] [in-biases hidden-biases out-biases]]
                     [in-act hidden-act out-act]]
  (map xpy
       (map mv (cons in-weights hidden-weights) (cons in-act (drop-last hidden-act)))
       (map trans-mv (conj hidden-weights out-weights) (conj (rest hidden-act) out-act))
       hidden-biases))

(defn awake-gibbs [params [in-act hidden-act out-act :as activations]]
  [in-act
   (map vsig (hidden-logits params activations))
   out-act])

(defn dream-gibbs [params activations]
  [(vsig (in-logits params activations))
   (map vsig (hidden-logits params activations))
   (softmax (out-logits params activations))])

(defn test-gibbs [params [in-act hidden-act out-act :as activations]]
  [in-act
   (map vsig (hidden-logits params activations))
   (softmax (out-logits params activations))])

(defn weight-diff [[in-act hidden-act out-act]]
  [(rk (first hidden-act) in-act)
   (map rk (rest hidden-act) (drop-last hidden-act))
   (rk out-act (last hidden-act))])

(defn func-power [f n]
  (apply comp (repeat n f)))

(defn map-params [f [in1 hidden1 out1] [in2 hidden2 out2]]
  [(f in1 in2) (map f hidden1 hidden2) (f out1 out2)])

(defn training-iteration [[[weights biases :as params] activations-list] index]
  (let [awake-activations ((func-power (partial awake-gibbs params) awake-iterations)
                            (activations-list index))
        dream-activations ((func-power (partial dream-gibbs params) dream-iterations)
                            awake-activations)
        weight-update (map-params #(scal learning-rate (xpy %1 (scal -1.0 %2)))
                                   (weight-diff awake-activations)
                                   (weight-diff dream-activations))
        bias-update (map-params #(scal learning-rate (xpy %1 (scal -1.0 %2)))
                                awake-activations
                                dream-activations)]
    [[(map-params xpy weights weight-update)
      (map-params xpy biases bias-update)]
     (assoc activations-list index awake-activations)]))

(defn training-epoch [training-data]
  (reduce training-iteration training-data (range num-training-examples)))

(defn test-csv-row [params [label-string & pixel-strings]]
  (let [label (Integer. label-string)
        pixels (map #(/ (Integer. %) 255.0) pixel-strings)
        [in-act hidden-act out-act] ((func-power (partial test-gibbs params) test-iterations)
                                      (test-activations pixels))]
    (= (imax out-act) label)))

(defn test-all [params]
  (println "testing...")
  (println (count (filter identity (map (partial test-csv-row params) (parse-csv (slurp "mnist_test.csv")))))))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println (count initial-activations))
  (-> [initial-params initial-activations] ((func-power training-epoch 20)) first test-all))

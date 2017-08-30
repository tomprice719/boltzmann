(ns boltzmann.core
  (:require [uncomplicate.neanderthal.core :refer [imax scal sum axpy rk entry! mv trans xpy axpby!]]
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
(def dream-iterations 2)
(def learning-rate 0.000001)
(def num-training-examples 60000) ;; actually 60000
(def input-weight-sd 0.03)
(def output-weight-sd 0.3)
(def test-iterations 10)
(def fisher-decay 0.99)

(defn map-rec [f c & args]
  (if (sequential? c)
    (doall (apply map (partial map-rec f) c args))
    (apply f c args)))

(def occasionally-counts (atom {}))

(defn get-or-initialize [key hash-atom default]
  (if-let [x (@hash-atom key)]
    x
    (do (swap! hash-atom #(assoc % key default))
        default)))

(defmacro occasionally [base & forms]
  `(let [[x# y#] (get-or-initialize
                 (quote key#) occasionally-counts
                 [(atom 0) (atom 1)])
         ~'count @x#]
     (if (> @x# @y#)
       (do ~@forms
           (swap! y# (partial * ~base))))
     (swap! x# inc)))

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

(def initial-activations-list (->> "mnist_train.csv"
                                   slurp
                                   parse-csv
                                   (take num-training-examples)
                                   (map activations-from-csv-row)))

(def initial-weights [(dge hidden-width input-size
                           (map #(* % input-weight-sd) (sample-normal (* hidden-width input-size))))
                      []
                      (dge 10 hidden-width
                           (map #(* % output-weight-sd) (sample-normal (* 10 hidden-width))))])

(def initial-biases [(dv input-size) [(dv hidden-width)] (dv 10)])

(def initial-params [initial-weights initial-biases])

(defn exp ^double [^double x]
  (Math/exp x))

(defn primitive-square ^double [^double x]
  (* x x))

(defn primitive-divide ^double [^double x ^double y]
  (/ x y ))

(def add-rec (partial map-rec axpy))
(def subtract-rec (partial map-rec #(axpy -1.0 %2 %1)))
(def square-rec (partial map-rec (fmap primitive-square)))
(def divide-rec (partial map-rec (fmap primitive-divide)))
(defn scal-rec [alpha x]
  (map-rec #(scal alpha %) x))

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

(defn weight-activations [[in-act hidden-act out-act]]
  [(rk (first hidden-act) in-act)
   (map rk (rest hidden-act) (drop-last hidden-act))
   (rk out-act (last hidden-act))])

(defn func-power [f n]
  (apply comp (repeat n f)))

(defn ema-update! [decay-factor new old]
  (axpby! (- 1 decay-factor) new decay-factor old))

(defn moments [[activations weight-activations]]
  [(square-rec weight-activations) weight-activations (square-rec activations) activations])

(defn ema-moments [total-activations old]
  (map-rec (partial ema-update! fisher-decay)
           (moments total-activations)
           old))

(defn weight-fisher-diagonal [[weight-second-moment weight-mean bias-second-moment bias-mean]]
  (subtract-rec weight-second-moment (square-rec weight-mean)))

(defn bias-fisher-diagonal [[weight-second-moment weight-mean bias-second-moment bias-mean]]
  (subtract-rec bias-second-moment (square-rec bias-mean)))

(defn updated-params [fisher-diagonal awake-activations dream-activations old]
  (add-rec
    (divide-rec (scal-rec learning-rate
                          (subtract-rec awake-activations dream-activations))
                fisher-diagonal)
    old))

(defn training-iteration [[[weights biases :as params]
                           moments
                           activations-acc]
                          next-activations]
  (let [awake-activations ((func-power (partial awake-gibbs params) awake-iterations)
                            next-activations)
        dream-activations ((func-power (partial dream-gibbs params) dream-iterations)
                            awake-activations)
        awake-weight-activations (weight-activations awake-activations)
        dream-weight-activations (weight-activations dream-activations)
        new-moments (ema-moments [dream-activations dream-weight-activations] moments)]
    (occasionally 1.5 (println count)
                  (println (weight-fisher-diagonal moments)))
    [[(updated-params (weight-fisher-diagonal new-moments)
                      awake-weight-activations dream-weight-activations weights)
      (updated-params (bias-fisher-diagonal new-moments)
                      awake-activations dream-activations biases)]
     new-moments
     (conj activations-acc awake-activations)]))

(defn inference->moments [params initial-activations]
  (let [awake-activations ((func-power (partial awake-gibbs params) awake-iterations)
                            initial-activations)
        dream-activations ((func-power (partial dream-gibbs params) dream-iterations)
                            awake-activations)
        dream-weight-activations (weight-activations dream-activations)]
    (moments [dream-activations dream-weight-activations])))

(def initial-moments
  (let [chunk-size 1000]
    (println "computing initial moments")
    (scal-rec (/ 1.0 chunk-size)
              (reduce add-rec
                      (map (partial inference->moments initial-params)
                           (take chunk-size initial-activations-list))))))

(defn training-epoch [training-data activations-list]
  (println "doing epoch")
  (reduce training-iteration training-data initial-activations-list))

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
  [& args]
  (-> [initial-params initial-moments []] (training-epoch initial-activations-list) first test-all))

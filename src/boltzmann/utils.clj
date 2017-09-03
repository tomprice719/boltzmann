(ns boltzmann.utils
  (:require [uncomplicate.neanderthal.core :refer [imax scal sum axpy rk entry! mv trans xpy axpby! amax]]
            [uncomplicate.neanderthal.native :refer [dv dge]]
            [uncomplicate.neanderthal.real :refer [entry]]
            [uncomplicate.fluokitten.core :refer [fmap]]))

(def ones (dv (repeat 10 1.0)))
(def eager-map (comp doall map))

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

(defn map-rec [f c & args]
  (if (sequential? c)
    (apply eager-map (partial map-rec f) c args)
    (apply f c args)))

(defn exp ^double [^double x]
  (Math/exp x))

(defn primitive-square ^double [^double x]
  (* x x))

(defn primitive-divide ^double [^double x ^double y]
  (/ x y ))

(defn primitive-multiply ^double [^double x ^double y]
  (* x y ))

(defn add-rec [x y] (map-rec axpy x y))
(defn subtract-rec [x y]  (map-rec #(axpy -1.0 %2 %1) x y))
(defn mult-rec [x y] (map-rec (fmap primitive-multiply) x y))
(defn square-rec [x] (mult-rec x x))
(defn divide-rec [x y]  (map-rec (fmap primitive-divide) x y))
(defn constant-scal-rec [alpha x]
  (map-rec #(scal alpha %) x))
(defn scal-rec [alpha x] (map-rec scal alpha x))

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

(defn func-power [f n]
  (apply comp (repeat n f)))

(defn unflatten [[head & tail]]
  [head (butlast tail) (last tail)])
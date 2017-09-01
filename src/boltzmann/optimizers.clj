(ns boltzmann.optimizers
  (:require
    [uncomplicate.neanderthal.core :refer [imax scal sum axpy rk entry! mv trans xpy axpby! amax]]
    [boltzmann.utils :refer :all]))

(def learning-rate 0.001)
(def fisher-decay 0.99)

(defprotocol optimizer
  (update-optimizer [this activations])
  (update-params [this params activations]))

(defn ema-update! [decay-factor new old]
  (axpby! (- 1 decay-factor) new decay-factor old))

(defn activations->moments [activations]
  (eager-map #(square-rec (apply subtract-rec %)) activations))

(defn opt-A-param-update [second-moment [awake-activations dream-activations] old]
  (add-rec
    (scal-rec
      (map-rec #(/ 1.0 (amax %)) second-moment)
      (constant-scal-rec learning-rate
                         (subtract-rec awake-activations dream-activations)))
    old))

(deftype opt-A [moments] optimizer
  (update-optimizer [this activations]
    (opt-A. (map-rec (partial ema-update! fisher-decay)
                    (activations->moments activations)
                    moments)))
  (update-params [this old-params activations]
    (eager-map opt-A-param-update moments activations old-params)))

(defn init-opt-A [activations-list]
  (opt-A.
    (constant-scal-rec (/ 1.0 (count activations-list))
                       (reduce add-rec
                               (eager-map activations->moments activations-list)))))

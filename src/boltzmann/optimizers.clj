(ns boltzmann.optimizers
  (:require
    [uncomplicate.neanderthal.core :refer [imax scal sum axpy rk entry! mv trans xpy axpby! amax]]
    [boltzmann.utils :refer :all]))

(def learning-rate 0.001)
(def fisher-decay 0.999)

(defprotocol optimizer
  (update-params-and-optimizer [this params activations]))

(defn weight-activations [[in-act hidden-act out-act]]
  [(rk (first hidden-act) in-act)
   (eager-map rk (rest hidden-act) (drop-last hidden-act))
   (rk out-act (last hidden-act))])

(defn ema-update! [decay-factor new old]
  (axpby! (- 1 decay-factor) new decay-factor old))

(defn activations->grad [[awake-unit-activations
                                    dream-unit-activations]]
  [(subtract-rec (weight-activations awake-unit-activations)
                 (weight-activations dream-unit-activations))
   (subtract-rec awake-unit-activations
                 dream-unit-activations)])

(defn opt-A-param-update [second-moment gradient old]
  (add-rec
    (scal-rec
      (map-rec #(/ 1.0 (amax %)) second-moment)
      (constant-scal-rec learning-rate
                         gradient))
    old))

(deftype opt-A [moments] optimizer
  (update-params-and-optimizer [this old-params activations]
    (let [grad (activations->grad activations)
          new-moments (map-rec (partial ema-update! fisher-decay)
                               (square-rec grad)
                               moments)]
      [(eager-map opt-A-param-update new-moments grad old-params)
       (opt-A. new-moments)])))

(defn init-opt-A [activations-list]
  (opt-A.
    (constant-scal-rec (/ 1.0 (count activations-list))
                       (reduce add-rec
                               (eager-map (comp square-rec activations->grad)
                                          activations-list)))))

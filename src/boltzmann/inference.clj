(ns boltzmann.inference
  (:require [boltzmann.utils :refer :all]
            [uncomplicate.neanderthal.core :refer [imax scal sum axpy rk entry! mv trans xpy axpby! amax]]))

(def awake-iterations 1)
(def dream-iterations 2)

(defn in-logits [[[[in-weights hidden-weights out-weights]
                   [in-biases hidden-biases out-biases]]
                  _]
                 [in-act hidden-act out-act]]
  (xpy in-biases (trans-mv in-weights (first hidden-act))))

(defn out-logits [[[[in-weights hidden-weights out-weights]
                    [in-biases hidden-biases out-biases]]
                   _]
                  [in-act hidden-act out-act]]
  (xpy out-biases (mv out-weights (last hidden-act))))

(defn hidden-logits [[[[in-weights hidden-weights out-weights]
                       [in-biases hidden-biases out-biases]]
                      _]
                     [in-act hidden-act out-act]]
  (eager-map xpy
             (eager-map mv
                        (cons in-weights hidden-weights)
                        (cons in-act (drop-last hidden-act)))
             (eager-map trans-mv
                        (conj hidden-weights out-weights)
                        (conj (rest hidden-act) out-act))
             hidden-biases))

(defn scaled-awake-gibbs [[_ [in-thickness hidden-thickness out-thickness] :as params]
                          [in-act hidden-act out-act :as activations]]
  [in-act
   (eager-map #(scal %1 (vsig %2)) hidden-thickness (hidden-logits params activations))
   out-act])

(defn scaled-dream-gibbs [[_ thickness :as params] activations]
  (scal-rec thickness
        [(vsig (in-logits params activations))
         (eager-map vsig (hidden-logits params activations))
         (softmax (out-logits params activations))]))

(defn scaled-test-gibbs [[_ [in-thickness hidden-thickness out-thickness] :as params]
                         [in-act hidden-act out-act :as activations]]
  [in-act
   (eager-map #(scal %1 (vsig %2)) hidden-thickness (hidden-logits params activations))
   (scal out-thickness (softmax (out-logits params activations)))])

(defn test-gibbs [[_ [in-thickness hidden-thickness out-thickness] :as params]
                         [in-act hidden-act out-act :as activations]]
  [in-act
   (eager-map vsig (hidden-logits params activations))
   (softmax (out-logits params activations))])

(defn infer-and-scale [params scaled-unit-activations]
  (let [scaled-awake-unit-activations ((func-power (partial scaled-awake-gibbs params) awake-iterations)
                                 scaled-unit-activations)
        scaled-dream-unit-activations ((func-power (partial scaled-dream-gibbs params) dream-iterations)
                                 scaled-awake-unit-activations)]
    [scaled-awake-unit-activations
     scaled-dream-unit-activations]))

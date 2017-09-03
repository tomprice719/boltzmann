(ns boltzmann.inference
  (:require [boltzmann.utils :refer :all]
            [uncomplicate.neanderthal.core :refer [imax scal sum axpy rk entry! mv trans xpy axpby! amax]]))

(def awake-iterations 1)
(def dream-iterations 2)

(defn in-logits [[[in-weights hidden-weights out-weights]
                  [in-biases hidden-biases out-biases]]
                 [in-act hidden-act out-act]]
  (xpy in-biases (trans-mv in-weights (first hidden-act))))

(defn out-logits [[[in-weights hidden-weights out-weights]
                   [in-biases hidden-biases out-biases]]
                  [in-act hidden-act out-act]]
  (xpy out-biases (mv out-weights (last hidden-act))))

(defn hidden-logits [[[in-weights hidden-weights out-weights]
                      [in-biases hidden-biases out-biases]]
                     [in-act hidden-act out-act]]
  (eager-map xpy
             (eager-map mv
                        (cons in-weights hidden-weights)
                        (cons in-act (drop-last hidden-act)))
             (eager-map trans-mv
                        (conj hidden-weights out-weights)
                        (conj (rest hidden-act) out-act))
             hidden-biases))

(defn output-probabilities [params activations]
  (softmax (out-logits params activations)))

(defn awake-gibbs [[in-thickness hidden-thickness out-thickness]
                          params
                          [in-act hidden-act out-act :as activations]]
  [in-act
   (eager-map #(scal %1 (vsig %2)) hidden-thickness (hidden-logits params activations))
   out-act])

(defn dream-gibbs [thickness params activations]
  (scal-rec thickness
        [(vsig (in-logits params activations))
         (eager-map vsig (hidden-logits params activations))
         (output-probabilities params activations)]))

(defn test-gibbs [[in-thickness hidden-thickness out-thickness]
                         params
                         [in-act hidden-act out-act :as activations]]
  [in-act
   (eager-map #(scal %1 (vsig %2)) hidden-thickness (hidden-logits params activations))
   (scal out-thickness (output-probabilities params activations))])

(defn infer [thickness params unit-activations]
  (let [awake-unit-activations ((func-power (partial awake-gibbs thickness params) awake-iterations)
                                 unit-activations)
        dream-unit-activations ((func-power (partial dream-gibbs thickness params) dream-iterations)
                                 awake-unit-activations)]
    [awake-unit-activations
     dream-unit-activations]))

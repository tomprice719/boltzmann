(defproject boltzmann "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [uncomplicate/neanderthal "0.14.0"]
                 [org.slf4j/slf4j-log4j12 "1.6.6"]
                 [uncomplicate/fluokitten "0.6.0"]
                 [clojure-csv/clojure-csv "2.0.1"]
                 [net.mikera/core.matrix "0.60.3"]]
  :main ^:skip-aot boltzmann.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})

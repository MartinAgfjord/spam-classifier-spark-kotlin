package se.agfjord.ml

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint

class SpamClassifier {

    private val model = createModel()

    private fun createModel(): LogisticRegressionModel {
        val conf = SparkConf().setAppName("email-spam").setMaster("local[*]")
        val sparkContext = JavaSparkContext(conf)
        val spam = sparkContext.textFile("./data/spam.txt", 4)
        val ham = sparkContext.textFile("./data/ham.txt", 4)

        val tf = HashingTF(10000)

        val spamFeatures  = spam.map { email : String -> tf.transform(email.split(" ")) }
        val normalFeatures  = ham.map { email : String -> tf.transform(email.split(" ")) }
        val positiveExamples = spamFeatures.map { features -> LabeledPoint(1.0, features) }
        val negativeExamples = normalFeatures.map { features -> LabeledPoint(0.0, features) }

        val trainingData = positiveExamples.union(negativeExamples)
        trainingData.cache()

        return LogisticRegressionWithSGD().run(trainingData.rdd())
    }

    fun predict(text: String): Prediction {
        val tf = HashingTF(10000)
        val vector = tf.transform(text.split(" "))
        val prediction = model.predict(vector)
        return when (prediction) {
            0.0 -> Prediction.HAM
            1.0 -> Prediction.SPAM
            else -> throw IllegalStateException("Expected model.predict to return either 0 or 1, but returned $prediction")

        }
    }

    enum class Prediction { HAM, SPAM }
}

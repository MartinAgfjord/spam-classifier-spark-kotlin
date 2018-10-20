package se.agfjord.ml

import org.hamcrest.CoreMatchers.`is`
import org.hamcrest.MatcherAssert.assertThat
import org.junit.Test
import kotlin.test.assertEquals

class SpamClassifierTest {

    companion object {
        private val underTest = SpamClassifier()
    }

    @Test
    fun `Predict spam`() {
        val prediction = underTest.predict("You are a WINNER! Go claim your prize")
        assertThat(prediction, `is`(SpamClassifier.Prediction.SPAM))
    }

    @Test
    fun `Predict ham`() {
        val prediction = underTest.predict("hi sorry yaar i forget tell you i cant come today")
        assertThat(prediction, `is`(SpamClassifier.Prediction.HAM))
    }

}

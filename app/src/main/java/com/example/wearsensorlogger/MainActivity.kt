package com.example.wearsensorlogger

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.sqrt

class MainActivity : AppCompatActivity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager

    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private var magnetometer: Sensor? = null

    private lateinit var playStopButton: Button
    private lateinit var statusText: TextView
    private lateinit var accMedianText: TextView
    private lateinit var gyrMedianText: TextView
    private lateinit var magMedianText: TextView

    private var isRecording = false
    private var recordingStartMillis: Long = 0L
    private var csvWriter: BufferedWriter? = null

    private var currentSecondEpoch: Long = -1L
    private val magnitudesBySensor = mutableMapOf<String, MutableList<Float>>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

        playStopButton = findViewById(R.id.playStopButton)
        statusText = findViewById(R.id.statusText)
        accMedianText = findViewById(R.id.accMedianText)
        gyrMedianText = findViewById(R.id.gyrMedianText)
        magMedianText = findViewById(R.id.magMedianText)

        updateMedianText("accelerometer", null)
        updateMedianText("gyroscope", null)
        updateMedianText("magnetometer", null)

        playStopButton.setOnClickListener {
            if (isRecording) {
                stopRecording()
            } else {
                startRecording()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        registerAvailableSensors()
        refreshAvailabilityUI()
    }

    override fun onPause() {
        super.onPause()
        unregisterAllSensors()
        if (isRecording) {
            stopRecording()
        } else {
            closeWriterSafely()
        }
    }

    override fun onDestroy() {
        unregisterAllSensors()
        closeWriterSafely()
        super.onDestroy()
    }

    private fun registerAvailableSensors() {
        accelerometer?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        gyroscope?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        magnetometer?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
    }

    private fun unregisterAllSensors() {
        sensorManager.unregisterListener(this)
    }

    private fun startRecording() {
        val start = System.currentTimeMillis()
        recordingStartMillis = start
        currentSecondEpoch = -1L
        magnitudesBySensor.clear()

        val filename = "sensor_log_${timestampForFilename(start)}.csv"
        val outputFile = File(filesDir, filename)

        try {
            csvWriter = BufferedWriter(FileWriter(outputFile, false)).apply {
                write("timestamp,sensor,value\n")
                flush()
            }

            isRecording = true
            playStopButton.text = getString(R.string.stop)
            statusText.text = getString(R.string.status_recording)
        } catch (_: IOException) {
            isRecording = false
            closeWriterSafely()
        }
    }

    private fun stopRecording() {
        flushSecondMedians(force = true)
        isRecording = false
        playStopButton.text = getString(R.string.play)
        statusText.text = getString(R.string.status_idle)
        closeWriterSafely()
    }

    private fun closeWriterSafely() {
        try {
            csvWriter?.flush()
            csvWriter?.close()
        } catch (_: IOException) {
            // Ignored intentionally.
        } finally {
            csvWriter = null
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        val sensorName = sensorTypeToName(event.sensor.type) ?: return
        val nowMillis = System.currentTimeMillis()
        val secondEpoch = nowMillis / 1000L

        if (currentSecondEpoch == -1L) {
            currentSecondEpoch = secondEpoch
        } else if (secondEpoch != currentSecondEpoch) {
            flushSecondMedians(force = false)
            currentSecondEpoch = secondEpoch
        }

        val x = event.values.getOrElse(0) { 0f }
        val y = event.values.getOrElse(1) { 0f }
        val z = event.values.getOrElse(2) { 0f }
        val magnitude = sqrt(x * x + y * y + z * z)

        val list = magnitudesBySensor.getOrPut(sensorName) { mutableListOf() }
        list.add(magnitude)

        if (isRecording) {
            appendCsvLine(nowMillis, sensorName, x, y, z)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    private fun appendCsvLine(timestampMillis: Long, sensor: String, x: Float, y: Float, z: Float) {
        val row = "$timestampMillis,$sensor,\"$x, $y, $z\"\n"
        try {
            csvWriter?.write(row)
        } catch (_: IOException) {
            stopRecording()
        }
    }

    private fun flushSecondMedians(force: Boolean) {
        val sensors = listOf("accelerometer", "gyroscope", "magnetometer")
        for (sensor in sensors) {
            val values = magnitudesBySensor[sensor]
            val median = if (values.isNullOrEmpty()) null else median(values)
            updateMedianText(sensor, median)
        }

        if (force || magnitudesBySensor.isNotEmpty()) {
            magnitudesBySensor.clear()
        }
    }

    private fun updateMedianText(sensor: String, median: Float?) {
        val value = when {
            !isSensorAvailable(sensor) -> getString(R.string.na)
            median == null -> "--"
            else -> String.format(Locale.US, "%.4f", median)
        }

        val sensorLabel = when (sensor) {
            "accelerometer" -> "Acelerómetro"
            "gyroscope" -> "Giroscopio"
            "magnetometer" -> "Magnetómetro"
            else -> sensor
        }

        val text = getString(R.string.median_format, sensorLabel, value)
        when (sensor) {
            "accelerometer" -> accMedianText.text = text
            "gyroscope" -> gyrMedianText.text = text
            "magnetometer" -> magMedianText.text = text
        }
    }

    private fun refreshAvailabilityUI() {
        updateMedianText("accelerometer", null)
        updateMedianText("gyroscope", null)
        updateMedianText("magnetometer", null)
    }

    private fun isSensorAvailable(sensor: String): Boolean {
        return when (sensor) {
            "accelerometer" -> accelerometer != null
            "gyroscope" -> gyroscope != null
            "magnetometer" -> magnetometer != null
            else -> false
        }
    }

    private fun sensorTypeToName(sensorType: Int): String? {
        return when (sensorType) {
            Sensor.TYPE_ACCELEROMETER -> "accelerometer"
            Sensor.TYPE_GYROSCOPE -> "gyroscope"
            Sensor.TYPE_MAGNETIC_FIELD -> "magnetometer"
            else -> null
        }
    }

    private fun median(values: List<Float>): Float {
        val sorted = values.sorted()
        val mid = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[mid - 1] + sorted[mid]) / 2f
        } else {
            sorted[mid]
        }
    }

    private fun timestampForFilename(timeMillis: Long): String {
        val formatter = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
        return formatter.format(Date(timeMillis))
    }
}

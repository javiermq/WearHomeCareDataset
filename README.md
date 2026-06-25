# WearSensorLogger (Wear OS)

Proyecto Wear OS en Kotlin + XML para registrar acelerómetro, giroscopio y magnetómetro en CSV.

## Nota sobre Gradle Wrapper
Este repositorio no incluye `gradle/wrapper/gradle-wrapper.jar` porque en este flujo no se admiten archivos binarios.

Para regenerarlo localmente:

```bash
gradle wrapper --gradle-version 8.7 --no-validate-url
```

Después podrás usar:

```bash
./gradlew assembleDebug
```


python model_internal_mag4d.py --scenes data2/scene1 --ablations MAG_4D UWB UWB_MAG_4D --models SIMPLE_LSTM CNN_2LSTM XGBOOST SVM --folds 5 --epochs 20
python model_internal_mag4d.py --scenes data1/scene1 --ablations MAG_4D UWB UWB_MAG_4D --models SIMPLE_LSTM CNN_2LSTM XGBOOST SVM --folds 5 --epochs 20
python model_internal_mag4d.py --scenes data3/sceneAB --ablations MAG_4D UWB UWB_MAG_4D --models SIMPLE_LSTM CNN_2LSTM XGBOOST SVM --folds 5 --epochs 20


python model_internal_imu7.py --scenes data2/scene1 --ablations IMU_7 UWB UWB_IMU_7 --models SIMPLE_LSTM CNN_2LSTM XGBOOST SVM --folds 5 --epochs 20
python model_internal_imu7.py --scenes data1/scene1 --ablations IMU_7 UWB UWB_IMU_7 --models SIMPLE_LSTM CNN_2LSTM XGBOOST SVM --folds 5 --epochs 20
python model_internal_imu7.py --scenes data3/sceneAB --ablations IMU_7 UWB UWB_IMU_7 --models SIMPLE_LSTM CNN_2LSTM XGBOOST SVM --folds 5 --epochs 20


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

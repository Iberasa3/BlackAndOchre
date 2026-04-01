import ee


class SM3Profiler:
    """
    Implementación de la estrategia SM3 (Environmental Pseudo-absence Selection).
    Utiliza One-Class SVM (OCSVM) para definir el hipervolumen del nicho ecológico
    y generar una máscara de 'similitud cero' con la que luego podremos generar las pseudoausencias.
    """

    def __init__(self, nu=0.1, gamma='auto'):
        """
        nu: Fracción aproximada de outliers (puntos de presencia que el modelo
            considerará fuera del nicho). Un valor de 0.1 (10%) suele ser estándar según los papers que he leído para evitar el overfitting.

        gamma: Parámetro de ancho del kernel RBF. 'auto' usa 1/n_features. Chequiar esto
        """
        self.nu = nu
        self.gamma = gamma
        self.trained_model = None




    def train_ocsvm(self, presence_fc, predictor_stack, scale=1000):
        """
        Entrena el OCSVM basándose únicamente en los datos de presencia.

        input:
            - presence_fc (ee.FeatureCollection): Conjunto de puntos de presencia
              (nidos/avistamientos) que deben contener una propiedad 'class' con valor 1.
            - predictor_stack (ee.Image): Stack multibanda con las variables ambientales
              (Golden Subset) que definen el espacio ecológico.
            - scale (int): Resolución espacial en metros para la extracción de datos
              (por defecto 1000m).

        output:
            - self.trained_model (ee.Classifier): Clasificador LIBSVM configurado y
              entrenado en modo ONE_CLASS, capaz de distinguir el nicho de los outliers.
        """

        training_samples = predictor_stack.sampleRegions(
            collection=presence_fc,
            properties=['class'],
            scale=scale,
            tileScale=16
        )

        # Configuramos el clasificador LIBSVM en modo ONE_CLASS
        # El parámetro 'oneClass' indica qué etiqueta representa la clase de interés
        self.trained_model = ee.Classifier.libsvm(
            svmType='ONE_CLASS',
            kernelType='RBF',
            nu=self.nu,
            gamma=None if self.gamma == 'auto' else self.gamma,
            oneClass=1
        ).train(
            features=training_samples,
            classProperty='class',
            inputProperties=predictor_stack.bandNames()
        )

        print("OCSVM Profiler entrenado con éxito.")
        return self.trained_model




    def get_zero_similarity_mask(self, predictor_stack, aoi):
        """
                Genera una máscara binaria de las zonas ambientalmente disimilares al nicho conocido.

                input:
                    - predictor_stack (ee.Image): El stack de variables ambientales (las mismas
                      usadas en el entrenamiento) que se clasificará para encontrar outliers.
                    - aoi (ee.Geometry): El área de interés (SM1_aoi) sobre la cual se proyectará
                      el modelo para identificar las zonas de pseudoausencia.

                output:
                    - zero_similarity_mask (ee.Image): Imagen binaria donde los píxeles con valor 1
                      representan las áreas de 'Similitud Cero' (outliers ambientales). Las áreas
                      similares al nicho quedan enmascaradas (transparentes/sin datos).
                """

        if not self.trained_model:
            raise Exception("El modelo debe ser entrenado antes de generar la máscara.")

        # Clasificamos el stack de variables
        # En ONE_CLASS, los outliers (zonas distintas) suelen recibir clase 0 o -1
        # y los valores dentro del nicho reciben clase 1.
        prediction = predictor_stack.clip(aoi).classify(self.trained_model)

        # Creamos la máscara, nos interesan solo las zonas que NO se parecen al nicho (0)
        zero_similarity_mask = prediction.eq(0)
        # selfMask() elimina los 0s geográficos para que la máscara sea solo el área apta para SM3
        return zero_similarity_mask.selfMask()
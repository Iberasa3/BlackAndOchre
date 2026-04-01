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
            considerará fuera del nicho). Un valor de 0.1 (10%) suele ser estándar según los papers que he leído.

        gamma: Parámetro de ancho del kernel RBF. 'auto' usa 1/n_features. Chequiar esto
        """
        self.nu = nu
        self.gamma = gamma
        self.trained_model = None

    def train_ocsvm(self, presence_fc, predictor_stack, scale=1000):
        """
        Entrena el OCSVM basándose únicamente en los datos de presencia.
        """
        # Extraemos los valores de las variables en los puntos de presencia
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
        Genera una máscara binaria donde:
        1 = Áreas de 'similitud cero' (ecológicamente muy distintas).
        0 = Áreas dentro del nicho o similares.
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
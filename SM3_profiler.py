import ee


class SM3Profiler:
    """
    Clase para implementar el Paso 2 del SM4 y el SM3 en su completitud
    Utiliza One-Class SVM para identificar zonas de 'similitud cero' respecto a las presencias
    """

    def __init__(self, nu=0.1, gamma='auto'):
        self.nu = nu  # Outlier fraction (standard 0.1 for invasive species)
        self.gamma = gamma
        self.trained_model = None

    def train_ocsvm(self, presence_fc, predictor_stack):
        """
        Trains the OCSVM using presence data.
        """
        # Sample predictors at presence locations
        training_samples = predictor_stack.sampleRegions(
            collection=presence_fc,
            properties=['class'],
            scale=1000,
            tileScale=16
        )

        # Train LIBSVM in ONE_CLASS mode. We set oneClass=1 because your gynes are labeled as 1.
        self.trained_model = ee.Classifier.libsvm(
            svmType='ONE_CLASS',
            kernelType='RBF',
            nu=self.nu,
            gamma=None if self.gamma == 'auto' else self.gamma,
            oneClass=1
        ).train(training_samples, 'class', predictor_stack.bandNames())

        return self.trained_model

    def get_zero_similarity_mask(self, predictor_stack, aoi):
        """
        Returns a mask where 1 represents zones with zero similarity to presences.
        """
        if not self.trained_model:
            raise Exception("Model not trained.")

        # Classify the AOI. 1 = Similarity, 0 = Zero Similarity (Outliers)
        similarity_map = predictor_stack.clip(aoi).classify(self.trained_model)

        # We isolate the '0' values as they represent environmentally hostile areas
        return similarity_map.eq(0).selfMask()


def generate_environmental_absences(presences, predictors, aoi, num_points, seed=67):
    """
    Función de utilidad para llamar desde el Notebook.
    """
    profiler = SM3Profiler()
    profiler.train_ocsvm(presences, predictors)
    mask = profiler.get_zero_similarity_mask(predictors, aoi)

    # Muestreo final de los 'ceros' inteligentes
    absences = mask.sample(
        region=aoi,
        scale=1000,
        numPixels=num_points * 2,  # Sobremuestreo para el truncamiento, esto es posible que haya que cambiarlo.
        seed=seed,
        geometries=True
    ).limit(num_points)

    return absences.map(lambda f: f.set('class', 0))

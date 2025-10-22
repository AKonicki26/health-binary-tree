interface TreeNode {
    feature_index: number;
    threshold: number;
    left: number;
    right: number;
    value: number[];
    is_leaf: boolean;
}

interface PreprocessingConfig {
    numeric: string[];
    categorical: string[];
    numeric_imputation: Record<string, number>;
    categorical_vocabulary: Record<string, any[]>;
    final_feature_order: Array<{
        source: string;
        kind: string;
        category?: any;
    }>;
}

interface ModelConfig {
    tree: {
        nodes: TreeNode[];
    };
    threshold: number;
    metrics: {
        precision: number;
        recall: number;
        f1: number;
        roc_auc: number;
    };
}

interface PredictionResult {
    probability: number;
    classification: 'HIGH' | 'LOW';
    threshold: number;
    metrics: {
        precision: number;
        recall: number;
        f1: number;
        roc_auc: number;
    };
    decisionPath: Array<{
        feature: string;
        value: number;
        threshold: number;
        direction: string;
    }>;
}

export class DecisionTreePredictor {
    private preproc: PreprocessingConfig | null = null;
    private model: ModelConfig | null = null;

    async initialize() {
        const [preprocResponse, modelResponse] = await Promise.all([
            fetch('/preproc_v1.json'),
            fetch('/dt_model_v1.json')
        ]);

        this.preproc = await preprocResponse.json();
        this.model = await modelResponse.json();
    }

    preprocessInput(userData: Record<string, number | string>): number[] {
        if (!this.preproc) throw new Error('Predictor not initialized');

        const features: number[] = [];

        // Process numeric features
        this.preproc.numeric.forEach(col => {
            let value = userData[col];
            if (value === undefined || value === '') {
                // Apply imputation
                value = this.preproc!.numeric_imputation[col];
            }
            features.push(Number(value));
        });

        // Process categorical features with one-hot encoding
        this.preproc.categorical.forEach(col => {
            const value = userData[col];
            const vocab = this.preproc!.categorical_vocabulary[col];

            // Create one-hot encoding
            vocab.forEach(category => {
                // Need to compare the actual values properly
                // Convert both to same type for comparison
                const userValue = Number(value);
                const categoryValue = Number(category);
                features.push(userValue === categoryValue ? 1 : 0);
            });
        });

        console.log('Preprocessed features:', features); // Debug
        console.log('Feature order:', this.preproc.final_feature_order); // Debug

        return features;
    }

    predict(userData: Record<string, number | string>): PredictionResult {
        if (!this.preproc || !this.model) {
            throw new Error('Predictor not initialized');
        }

        // Preprocess input
        const features = this.preprocessInput(userData);

        // Trace through tree and collect path
        const decisionPath: PredictionResult['decisionPath'] = [];
        let nodeIdx = 0;

        while (!this.model.tree.nodes[nodeIdx].is_leaf) {
            const node = this.model.tree.nodes[nodeIdx];
            const featureIdx = node.feature_index;
            const featureInfo = this.preproc.final_feature_order[featureIdx];
            const featureValue = features[featureIdx];
            const threshold = node.threshold;

            let featureName: string;
            if (featureInfo.kind === 'numeric') {
                featureName = featureInfo.source;
            } else {
                featureName = `${featureInfo.source}=${featureInfo.category}`;
            }

            const direction = featureValue > threshold ? 'right' : 'left';

            decisionPath.push({
                feature: featureName,
                value: featureValue,
                threshold: threshold,
                direction: direction
            });

            nodeIdx = featureValue > threshold ? node.right : node.left;
        }

        // Get prediction from leaf node
        const leafNode = this.model.tree.nodes[nodeIdx];
        const leafValues = leafNode.value;
        const total = leafValues.reduce((sum, val) => sum + val, 0);
        const probability = total > 0 ? leafValues[1] / total : 0;

        const classification = probability >= this.model.threshold ? 'HIGH' : 'LOW';

        return {
            probability,
            classification,
            threshold: this.model.threshold,
            metrics: this.model.metrics,
            decisionPath
        };
    }
}
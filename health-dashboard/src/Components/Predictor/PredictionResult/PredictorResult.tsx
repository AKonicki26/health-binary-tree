import React from 'react';
import './PredictorResult.css';

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

interface PredictorResultsProps {
    result: PredictionResult;
    inputData: Record<string, number | string>;
    onReset: () => void;
}

const PredictorResults: React.FC<PredictorResultsProps> = ({ result, inputData, onReset }) => {
    const getRiskColor = () => {
        return result.classification === 'HIGH' ? '#ef4444' : '#22c55e';
    };

    return (
        <div className="predictor-results">
            <div className="results-header">
                <h2>Prediction Results</h2>
                <button onClick={onReset} className="reset-button">
                    New Assessment
                </button>
            </div>

            <div className="results-grid">
                {/* Main Prediction */}
                <div className="result-card main-result" style={{ borderColor: getRiskColor() }}>
                    <h3>Risk Classification</h3>
                    <div className="risk-badge" style={{ backgroundColor: getRiskColor() }}>
                        {result.classification} RISK
                    </div>
                    <div className="probability">
                        <span className="probability-label">Probability:</span>
                        <span className="probability-value">
                            {(result.probability * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div className="threshold-info">
                        Threshold: {(result.threshold * 100).toFixed(1)}%
                    </div>
                </div>

                {/* Model Performance */}
                <div className="result-card">
                    <h3>Model Performance</h3>
                    <div className="metrics-list">
                        <div className="metric">
                            <span>Precision:</span>
                            <strong>{(result.metrics.precision * 100).toFixed(1)}%</strong>
                        </div>
                        <div className="metric">
                            <span>Recall:</span>
                            <strong>{(result.metrics.recall * 100).toFixed(1)}%</strong>
                        </div>
                        <div className="metric">
                            <span>F1 Score:</span>
                            <strong>{(result.metrics.f1 * 100).toFixed(1)}%</strong>
                        </div>
                        <div className="metric">
                            <span>ROC AUC:</span>
                            <strong>{result.metrics.roc_auc.toFixed(3)}</strong>
                        </div>
                    </div>
                </div>

                {/* Input Summary */}
                <div className="result-card">
                    <h3>Input Summary</h3>
                    <div className="input-summary">
                        {inputData && Object.keys(inputData).length > 0 ? (
                            Object.entries(inputData).map(([key, value]) => (
                                <div key={key} className="input-item">
                                    <span className="input-label">
                                        {key.replace(/_/g, ' ')}:
                                    </span>
                                    <span className="input-value">{String(value)}</span>
                                </div>
                            ))
                        ) : (
                            <div className="input-item">
                                <span className="input-label">No input data available</span>
                            </div>
                        )}
                    </div>
                </div>

                {/* Decision Path */}
                <div className="result-card decision-path-card">
                    <h3>Decision Path</h3>
                    <div className="decision-path">
                        {result.decisionPath.map((step, index) => (
                            <div key={index} className="path-step">
                                <div className="step-number">{index + 1}</div>
                                <div className="step-content">
                                    <div className="step-feature">{step.feature}</div>
                                    <div className="step-condition">
                                        {step.value.toFixed(2)} {step.direction === 'left' ? '≤' : '>'} {step.threshold.toFixed(2)}
                                    </div>
                                    <div className="step-arrow">→ Go {step.direction}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="disclaimer">
                <strong>⚠️ DISCLAIMER:</strong> This is for educational purposes only.
                Please consult healthcare professionals for medical advice.
            </div>
        </div>
    );
};

export default PredictorResults;
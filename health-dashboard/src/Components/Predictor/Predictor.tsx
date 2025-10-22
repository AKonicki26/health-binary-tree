import React, { useState, useEffect } from 'react';
import PredictorForm from './PredictorForm/PredictorForm';
import PredictorResults from './PredictionResult/PredictorResult';
import { DecisionTreePredictor } from './PredictionEngine';
import './Predictor.css';

const Predictor: React.FC = () => {
    const [predictor, setPredictor] = useState<DecisionTreePredictor | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<any | null>(null);
    const [inputData, setInputData] = useState<Record<string, number | string> | null>(null);

    useEffect(() => {
        const initPredictor = async () => {
            try {
                const pred = new DecisionTreePredictor();
                await pred.initialize();
                setPredictor(pred);
                setLoading(false);
            } catch (err) {
                console.error('Error initializing predictor:', err);
                setError('Failed to load prediction model. Please try again later.');
                setLoading(false);
            }
        };

        initPredictor();
    }, []);

    const handleSubmit = (data: Record<string, number | string>) => {
        if (!predictor) return;

        try {
            console.log('Raw input data received:', data); // Debug: see what data we received
            console.log('Number of fields:', Object.keys(data).length); // Debug: count fields
            
            const prediction = predictor.predict(data);
            console.log('Prediction result:', prediction); // Debug: see prediction
            
            setResult(prediction);
            setInputData(data);  // Store the original input for display
        } catch (err) {
            console.error('Error making prediction:', err);
            setError('Failed to make prediction. Please check your input and try again.');
        }
    };

    const handleReset = () => {
        setResult(null);
        setInputData(null);
    };

    if (loading) {
        return (
            <div className="predictor-container">
                <div className="predictor-loading">
                    <div className="spinner"></div>
                    <p>Loading prediction model...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="predictor-container">
                <div className="predictor-error">
                    <h3>Error</h3>
                    <p>{error}</p>
                    <button onClick={() => window.location.reload()}>
                        Reload Page
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="predictor-container">
            {!result ? (
                <PredictorForm onSubmit={handleSubmit} />
            ) : (
                <PredictorResults
                    result={result}
                    inputData={inputData!}
                    onReset={handleReset}
                />
            )}
        </div>
    );
};

export default Predictor;
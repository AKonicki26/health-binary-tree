import './PredictorForm.css';
import React, { useState, useEffect } from 'react';

interface Feature {
    name: string;
    type: 'numeric' | 'categorical' | 'binary';
    min?: number;
    max?: number;
    options?: (string | number)[];
    default?: number | string;
}

interface PredictorFormProps {
    onSubmit: (data: Record<string, number | string>) => void;
}

const PredictorForm: React.FC<PredictorFormProps> = ({ onSubmit }) => {
    const [features, setFeatures] = useState<Feature[]>([]);
    const [formData, setFormData] = useState<Record<string, number | string>>({});
    const [loading, setLoading] = useState(true);
    const [gender, setGender] = useState<string>('');

    useEffect(() => {
        // Load preprocessing config to build form
        fetch('/preproc_v1.json')
            .then(res => res.json())
            .then(data => {
                const featureList: Feature[] = [];
                
                // Add numeric features
                data.numeric.forEach((name: string) => {
                    featureList.push({
                        name,
                        type: 'numeric',
                        min: data.numeric_ranges_train[name].min,
                        max: data.numeric_ranges_train[name].max,
                        default: data.numeric_imputation[name]
                    });
                });
                
                // Add categorical features
                data.categorical.forEach((name: string) => {
                    const options = data.categorical_vocabulary[name].filter((v: any) => v !== null);
                    
                    // Skip gender-related fields (we'll handle them separately)
                    if (name === 'Is_Male' || name === 'Is_Female' || name === 'Gender_Other') {
                        return;
                    }
                    
                    // Check if it's a binary field (only has 0 and 1 as options)
                    const isBinary = options.length === 2 && 
                                    options.includes(0) && 
                                    options.includes(1);
                    
                    featureList.push({
                        name,
                        type: isBinary ? 'binary' : 'categorical',
                        options: isBinary ? undefined : options
                    });
                });
                
                setFeatures(featureList);
                setLoading(false);
            })
            .catch(err => {
                console.error('Error loading preprocessing config:', err);
                setLoading(false);
            });
    }, []);

    const handleInputChange = (name: string, value: string | number) => {
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleCheckboxChange = (name: string, checked: boolean) => {
        setFormData(prev => ({
            ...prev,
            [name]: checked ? 1 : 0
        }));
    };

    const handleGenderChange = (value: string) => {
        setGender(value);
        // Set the appropriate gender flags
        setFormData(prev => ({
            ...prev,
            Is_Male: value === 'Male' ? 1 : 0,
            Is_Female: value === 'Female' ? 1 : 0,
            Gender_Other: value === 'Other' ? 1 : 0
        }));
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        
        // Ensure all fields have values (use defaults if needed)
        const completeData = { ...formData };
        
        // Add any missing numeric fields with their defaults
        features.filter(f => f.type === 'numeric').forEach(feature => {
            if (completeData[feature.name] === undefined || completeData[feature.name] === '') {
                completeData[feature.name] = feature.default ?? 0;
            }
        });
        
        // Add any missing binary fields (default to 0)
        features.filter(f => f.type === 'binary').forEach(feature => {
            if (completeData[feature.name] === undefined) {
                completeData[feature.name] = 0;
            }
        });
        
        // Ensure gender fields are set
        if (!completeData.Is_Male) completeData.Is_Male = 0;
        if (!completeData.Is_Female) completeData.Is_Female = 0;
        if (!completeData.Gender_Other) completeData.Gender_Other = 0;
        
        console.log('Form submitting with data:', completeData); // Debug
        console.log('Total fields:', Object.keys(completeData).length); // Debug
        
        onSubmit(completeData);
    };

    if (loading) {
        return <div className="predictor-form-loading">Loading form...</div>;
    }

    return (
        <form className="predictor-form" onSubmit={handleSubmit}>
            <h2>COVID-19 Risk Assessment</h2>
            <p className="form-description">
                Enter patient information to assess COVID-19 risk. Leave fields blank to use default values.
            </p>

            {/* Gender Selection */}
            <div className="form-section">
                <h3>Personal Information</h3>
                <div className="form-field">
                    <label htmlFor="gender">Gender</label>
                    <select
                        id="gender"
                        name="gender"
                        value={gender}
                        onChange={(e) => handleGenderChange(e.target.value)}
                        required
                    >
                        <option value="">-- Select Gender --</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                        <option value="Other">Other</option>
                    </select>
                </div>
            </div>

            {/* Numeric Features */}
            <div className="form-section">
                <h3>Vital Signs</h3>
                <div className="form-grid">
                    {features.filter(f => f.type === 'numeric').map(feature => (
                        <div key={feature.name} className="form-field">
                            <label htmlFor={feature.name}>
                                {feature.name.replace(/_/g, ' ')}
                                {feature.min !== undefined && (
                                    <span className="field-hint">
                                        ({feature.min.toFixed(0)} - {feature.max!.toFixed(0)})
                                    </span>
                                )}
                            </label>
                            <input
                                type="number"
                                id={feature.name}
                                name={feature.name}
                                min={feature.min}
                                max={feature.max}
                                step="0.1"
                                value={formData[feature.name] ?? ''}
                                onChange={(e) => handleInputChange(feature.name, e.target.value)}
                                placeholder={`Default: ${Number(feature.default ?? 0).toFixed(1)}`}
                            />
                        </div>
                    ))}
                </div>
            </div>

            {/* Binary Features (Checkboxes) */}
            <div className="form-section">
                <h3>Symptoms & Conditions</h3>
                <div className="checkbox-grid">
                    {features.filter(f => f.type === 'binary').map(feature => (
                        <div key={feature.name} className="checkbox-field">
                            <label>
                                <input
                                    type="checkbox"
                                    name={feature.name}
                                    checked={formData[feature.name] === 1}
                                    onChange={(e) => handleCheckboxChange(feature.name, e.target.checked)}
                                />
                                <span>{feature.name.replace(/_/g, ' ')}</span>
                            </label>
                        </div>
                    ))}
                </div>
            </div>

            {/* Non-binary Categorical Features (if any remain) */}
            {features.filter(f => f.type === 'categorical').length > 0 && (
                <div className="form-section">
                    <h3>Additional Information</h3>
                    <div className="form-grid">
                        {features.filter(f => f.type === 'categorical').map(feature => (
                            <div key={feature.name} className="form-field">
                                <label htmlFor={feature.name}>
                                    {feature.name.replace(/_/g, ' ')}
                                </label>
                                <select
                                    id={feature.name}
                                    name={feature.name}
                                    value={formData[feature.name] ?? ''}
                                    onChange={(e) => handleInputChange(feature.name, e.target.value)}
                                >
                                    <option value="">-- Select --</option>
                                    {feature.options?.map(option => (
                                        <option key={option} value={option}>
                                            {option}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <button type="submit" className="submit-button">
                Calculate Risk
            </button>
        </form>
    );
};

export default PredictorForm;
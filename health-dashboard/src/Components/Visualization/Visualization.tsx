import React from 'react';
import './Visualization.css';

interface VisualizationProps {
    imagePath: string;
    title?: string;
    description?: string;
}

const Visualization: React.FC<VisualizationProps> = ({ imagePath, title, description }) => {
    return (
        <div className="visualization-container">
            {title && <h2 className="visualization-title">{title}</h2>}
            {description && <p className="visualization-description">{description}</p>}
            <div className="visualization-image-wrapper">
                <img
                    src={imagePath}
                    alt={title || "Visualization"}
                    className="visualization-image"
                />
            </div>
        </div>
    );
};

export default Visualization;
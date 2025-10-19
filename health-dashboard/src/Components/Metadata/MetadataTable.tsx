
import React, { useEffect, useState } from 'react';
import './MetadataTable.css';

interface MetadataTableProps {
    csvPath: string;
    title?: string;
}

const MetadataTable: React.FC<MetadataTableProps> = ({ csvPath, title }) => {
    const [tableData, setTableData] = useState<string[][]>([]);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState<boolean>(true);

    useEffect(() => {
        const fetchAndParseCSV = async () => {
            try {
                setLoading(true);
                const response = await fetch(csvPath);

                if (!response.ok) {
                    throw new Error(`Failed to fetch CSV from ${csvPath}`);
                }

                const text = await response.text();
                const lines = text.trim().split('\n');
                const data = lines.map(line => line.split(','));

                setTableData(data);
                setError(null);
            } catch (err) {
                console.error(err);
                setError(`Error loading CSV from ${csvPath}`);
            } finally {
                setLoading(false);
            }
        };

        fetchAndParseCSV();
    }, [csvPath]);

    if (loading) {
        return <div className="metadata-table-container">Loading...</div>;
    }

    if (error) {
        return <div className="metadata-table-container error">{error}</div>;
    }

    return (
        <div className="metadata-table-container">
            {title && <h2 className="metadata-table-title">{title}</h2>}
            <table className="metadata-table">
                <thead>
                {tableData.length > 0 && (
                    <tr>
                        {tableData[0].map((cell, index) => (
                            <th key={index}>{cell}</th>
                        ))}
                    </tr>
                )}
                </thead>
                <tbody>
                {tableData.slice(1).map((row, rowIndex) => (
                    <tr key={rowIndex}>
                        {row.map((cell, cellIndex) => (
                            <td key={cellIndex}>{cell}</td>
                        ))}
                    </tr>
                ))}
                </tbody>
            </table>
        </div>
    );
};

export default MetadataTable;
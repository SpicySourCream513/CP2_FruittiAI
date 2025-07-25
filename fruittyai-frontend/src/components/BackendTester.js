import React, { useState } from 'react';

const BackendTester = () => {
  const [testResults, setTestResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isVisible, setIsVisible] = useState(false);

  const testBackendConnection = async () => {
    setIsLoading(true);
    const results = {
      api: { status: 'testing...', message: '' },
      models: { status: 'testing...', message: '' }
    };
    setTestResults({ ...results });

    try {
      // Test 1: Basic API connection
      const apiResponse = await fetch('http://localhost:8000/');
      const apiData = await apiResponse.json();
      results.api = { 
        status: apiResponse.ok ? 'success' : 'error', 
        message: apiData.message || 'API connected' 
      };
      setTestResults({ ...results });

      // Test 2: Models status (NEW endpoint)
      const modelsResponse = await fetch('http://localhost:8000/models/status');
      const modelsData = await modelsResponse.json();
      results.models = { 
        status: modelsData.models_loaded ? 'success' : 'error', 
        message: modelsData.message 
      };
      setTestResults({ ...results });

    } catch (error) {
      results.api = { status: 'error', message: `Connection failed: ${error.message}` };
      results.models = { status: 'error', message: 'Could not test - API unavailable' };
      setTestResults({ ...results });
    }
    
    setIsLoading(false);
  };

  const getStatusStyle = (status) => ({
    padding: '3px 6px',
    borderRadius: '3px',
    fontSize: '10px',
    fontWeight: '600',
    color: 'white',
    background: status === 'success' ? '#28a745' : 
                status === 'error' ? '#dc3545' : '#6c757d'
  });

  return (
    <>
      {/* Toggle Button */}
      <button
        onClick={() => setIsVisible(!isVisible)}
        style={{
          position: 'fixed',
          top: '80px',
          right: '20px',
          background: '#007bff',
          color: 'white',
          border: 'none',
          borderRadius: '50%',
          width: '40px',
          height: '40px',
          cursor: 'pointer',
          zIndex: 1001,
          fontSize: '16px',
          boxShadow: '0 2px 10px rgba(0,0,0,0.2)'
        }}
        title="Backend Status"
      >
        🔧
      </button>

      {/* Test Panel */}
      {isVisible && (
        <div style={{
          position: 'fixed',
          top: '130px',
          right: '20px',
          background: 'white',
          padding: '15px',
          borderRadius: '8px',
          boxShadow: '0 4px 20px rgba(0,0,0,0.15)',
          zIndex: 1000,
          maxWidth: '250px',
          border: '1px solid #ddd'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
            <h4 style={{ margin: '0', color: '#333', fontSize: '14px' }}>Backend Status</h4>
            <button 
              onClick={() => setIsVisible(false)}
              style={{ background: 'none', border: 'none', fontSize: '16px', cursor: 'pointer' }}
            >
              ✕
            </button>
          </div>
          
          <button 
            onClick={testBackendConnection}
            disabled={isLoading}
            style={{
              width: '100%',
              padding: '8px',
              background: '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: isLoading ? 'not-allowed' : 'pointer',
              marginBottom: '10px',
              fontSize: '12px'
            }}
          >
            {isLoading ? 'Testing...' : 'Test Connection'}
          </button>

          {testResults && (
            <div style={{ fontSize: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '5px' }}>
                <span>API:</span>
                <span style={getStatusStyle(testResults.api.status)}>
                  {testResults.api.status}
                </span>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>Models:</span>
                <span style={getStatusStyle(testResults.models.status)}>
                  {testResults.models.status}
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </>
  );
};

export default BackendTester;
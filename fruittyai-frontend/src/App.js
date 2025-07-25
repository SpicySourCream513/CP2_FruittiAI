import React, { useState } from 'react';
import HomePage from './components/HomePage';
import GradingSystem from './components/GradingSystem';

// Main App Component
const App = () => {
  const [currentPage, setCurrentPage] = useState('home');

  return (
    <div>
      {currentPage === 'home' ? (
        <HomePage onNavigateToGrading={() => setCurrentPage('grading')} />
      ) : (
        <GradingSystem onBackToHome={() => setCurrentPage('home')} />
      )}
    </div>
  );
};

export default App;
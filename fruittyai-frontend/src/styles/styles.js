// src/styles/styles.js
// Shared styles for all components

export const styles = {
  // Global styles
  app: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    margin: 0,
    padding: 0,
    lineHeight: '1.6',
    color: '#333',
  },

  // Navigation Bar
  navbar: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    background: '#0a5247',
    padding: '5px 0',
    zIndex: 1000,
    boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
  },

  navContent: {
    maxWidth: '50px',
    margin: '0 auto',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '0 20px',
    fontFamily: 'Leckerli One, cursive', 
  },

  navLogo: {
    fontSize: '1.4rem',
    fontWeight: '500',
    color: '#a0e34c',
    textDecoration: 'none',
    letterSpacing: '1px',
  },

  // Buttons
  btn: {
    padding: '12px 24px',
    border: 'none',
    borderRadius: '8px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '14px',
  },
  
  btnPrimary: {
    background: 'linear-gradient(135deg, #007bff, #0056b3)',
    color: 'white',
  },
  
  btnSuccess: {
    background: 'linear-gradient(135deg, #28a745, #1e7e34)',
    color: 'white',
  },

  ctaButton: {
    display: 'inline-block',
    padding: '18px 40px',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    textDecoration: 'none',
    borderRadius: '50px',
    fontWeight: '600',
    fontSize: '1.1rem',
    border: 'none',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    boxShadow: '0 10px 30px rgba(102, 126, 234, 0.3)',
  },

  // Add all other styles here...
  // (You can move the rest from your individual component files)
};
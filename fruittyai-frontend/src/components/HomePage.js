import React from 'react';
import BackendTester from './BackendTester';


const styles = {
  app: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    margin: 0,
    padding: 0,
    lineHeight: '1.6',
    color: '#333',
  },

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

  hero: {
    height: '80vh',
    background: 'linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)), url("/images/background1.jpg") center/cover',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    textAlign: 'center',
    color: 'white',
    position: 'relative',
    paddingTop: '80px'
  },
  
  heroTitle: {
    fontSize: '4.5rem',
    fontWeight: '800',
    marginBottom: '1rem',
    textShadow: '2px 2px 4px rgba(0,0,0,0.5)',
    animation: 'fadeInUp 1s ease-out',
    fontFamily: 'Leckerli One", cursive',
  },
  
  heroSubtitle: {
    fontSize: '1.5rem',
    fontWeight: '300',
    marginBottom: '2rem',
    opacity: 0.9,
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

  section: {
    padding: '100px 20px',
  },
  
  container: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '0 20px',
  },
  
  sectionTitle: {
    textAlign: 'center',
    fontSize: '2.5rem',
    color: '#2c3e50',
    marginBottom: '30px',
    fontWeight: '700',
  },
  
  introSection: {
    padding: '100px 20px',
    background: '#f8f9fa',
  },
  
  introContent: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '60px',
    alignItems: 'center',
    maxWidth: '1200px',
    margin: '0 auto',
  },
  
  introText: {
    fontSize: '1.1rem',
    color: '#666',
    lineHeight: '1.8',
  },
  
  introTitle: {
    fontSize: '3rem',
    color: '#2c3e50',
    marginBottom: '1rem',
    fontWeight: '700',
  },
  
  introSubtitle: {
    fontSize: '1.3rem',
    color: '#667eea',
    marginBottom: '2rem',
    fontWeight: '600',
  },

  howItWorks: {
    padding: '40px 20px',
    background: 'white',
  },
  
  stepsContainer: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '40px',
    maxWidth: '1000px',
    margin: '0 auto',
  },
  
  step: {
    textAlign: 'center',
    padding: '40px 20px',
    borderRadius: '20px',
    background: '#f8f9fa',
    transition: 'transform 0.3s ease, box-shadow 0.3s ease',
  },
  
  stepIcon: {
    width: '80px',
    height: '80px',
    margin: '0 auto 30px',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '2rem',
    color: 'white',
  },

  demoSection: {
    padding: '30px 20px',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    textAlign: 'center',
  },
  
  videoContainer: {
    maxWidth: '1200px',
    margin: '0 auto',
    position: 'relative',
    paddingBottom: '56.25%',
    height: '0',
    borderRadius: '10px',
    overflow: 'hidden',
    boxShadow: '0 20px 40px rgba(0,0,0,0.3)',
  },

  behindScenes: {
    padding: '40px 70px',
    background: '#f8f9fa',
    textAlign: 'center',
  },
  
  processGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: '30px',
    maxWidth: '1200px',
    margin: '0 auto',
  },
  
  processBox: {
    background: 'white',
    padding: '30px',
    borderRadius: '15px',
    boxShadow: '0 10px 30px rgba(0,0,0,0.1)',
    transition: 'transform 0.3s ease, box-shadow 0.3s ease',
    borderLeft: '4px solid #667eea',
  },

  startNowSection: {
    padding: '15px 15px',
    background: 'linear-gradient(135deg, #2c3e50 0%, #34495e 100%)',
    textAlign: 'center',
    color: 'white',
  },
};

const Navigation = () => {
  return (
    <nav style={styles.navbar}>
      <div style={styles.navContent}>
        <div style={styles.navLogo}>Fruitti.AI</div>
      </div>
    </nav>
  );
};

const HomePage = ({ onNavigateToGrading }) => {
  return (
    <div style={styles.app}>
      <Navigation />
      <BackendTester />
      
      {/* Hero Section */}
      <section style={styles.hero}>
        <div>
          <h1 style={styles.heroTitle}>Struggle to pick <br/>the best fruit?</h1>
          <h2 style={styles.heroSubtitle}>AI-Powered Fruit Classification & Grading System</h2>
          <button 
            onClick={onNavigateToGrading}
            style={{
              ...styles.ctaButton,
              padding: '20px 50px',
              fontSize: '1.2rem',
            }}
          >
            Start Now
          </button>
        </div>
      </section>

      {/* Introduction Section */}
      <section id="intro" style={styles.introSection}>
        <div style={styles.introContent}>
          <div>
            <h2 style={styles.introTitle}>Smart Fruit <br /> Smart Choices</h2>
            <h3 style={styles.introSubtitle}>Our Philosophy</h3>
            <p style={styles.introText}>
              We believe that technology should enhance agricultural practices and food quality assessment. 
              FruittyAI combines cutting-edge artificial intelligence with computer vision to revolutionize 
              how we classify and grade fruits. Our system provides instant, accurate, and consistent quality 
              assessment that helps farmers, distributors, and retailers make informed decisions about their produce.
            </p>
          </div>
          <div style={{ textAlign: 'center' }}>
            <img 
              src="/images/applesingrocery.jpg" 
              alt="Fresh fruits"
              style={{ borderRadius: '20px', maxWidth: '100%', boxShadow: '0 20px 40px rgba(0,0,0,0.1)' }}
            />
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section style={styles.howItWorks}>
        <div style={styles.container}>
          <h2 style={styles.sectionTitle}>How It Works?</h2>
          <div style={styles.stepsContainer}>
            <div style={styles.step}>
              <div style={{...styles.stepIcon, background: 'linear-gradient(135deg, #667eea, #764ba2)'}}>
                📸
              </div>
              <h3 style={{ fontSize: '1.5rem', color: '#2c3e50', marginBottom: '15px', fontWeight: '600' }}>Detect</h3>
              <p style={{ color: '#666', lineHeight: '1.6' }}>
                Simply open your camera or upload an image of your fruit. Our system instantly detects 
                and focuses on the fruit in the frame.
              </p>
            </div>
            <div style={styles.step}>
              <div style={{...styles.stepIcon, background: 'linear-gradient(135deg, #f093fb, #f5576c)'}}>
                🧠
              </div>
              <h3 style={{ fontSize: '1.5rem', color: '#2c3e50', marginBottom: '15px', fontWeight: '600' }}>Analyze</h3>
              <p style={{ color: '#666', lineHeight: '1.6' }}>
                Advanced deep learning algorithms process the image, analyzing color, texture, size, 
                ripeness, and potential defects in seconds.
              </p>
            </div>
            <div style={styles.step}>
              <div style={{...styles.stepIcon, background: 'linear-gradient(135deg, #4facfe, #00f2fe)'}}>
                ✅
              </div>
              <h3 style={{ fontSize: '1.5rem', color: '#2c3e50', marginBottom: '15px', fontWeight: '600' }}>Done</h3>
              <p style={{ color: '#666', lineHeight: '1.6' }}>
                Receive comprehensive grading results including quality score, ripeness level, 
                defect analysis, and market classification.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Demo Video Section */}
      <section style={styles.demoSection}>
        <div style={styles.container}>
          <h2 style={{...styles.sectionTitle, color: 'white', marginBottom: '50px'}}>Demo Video</h2>
          <div style={styles.videoContainer}>
            <iframe 
              style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
              src="https://www.youtube.com/embed/aJ5alwLUNzg"
              frameBorder="0" 
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
              allowFullScreen
            />
          </div>
        </div>
      </section>

      {/* Behind the Scenes Section */}
      <section style={styles.behindScenes}>
        <div style={styles.container}>
          <h2 style={styles.sectionTitle}>Behind the Scenes</h2>
          <p style={{ textAlign: 'center', maxWidth: '800px', margin: '0 auto 60px', fontSize: '1.1rem', color: '#666', lineHeight: '1.8' }}>
            Our AI system operates through a sophisticated pipeline of interconnected processes, 
            each optimized for accuracy and speed.
          </p>
          <div style={styles.processGrid}>
            {[
              {
                title: "Image Acquisition",
                description: "High-resolution camera capture with automatic focus adjustment and optimal lighting detection for consistent image quality."
              },
              {
                title: "Preprocessing",
                description: "Advanced image enhancement including noise reduction, contrast optimization, and color normalization to prepare images for accurate analysis."
              },
              {
                title: "Object Detection",
                description: "YOLO-based neural networks identify and isolate fruits within the frame, filtering out background elements."
              },
              {
                title: "Feature Extraction",
                description: "Convolutional neural networks analyze color patterns, texture details, shape characteristics, and size measurements."
              },
              {
                title: "Classification Model",
                description: "Deep learning models trained on over 100,000 fruit images classify type, variety, ripeness stage, and quality parameters."
              },
              {
                title: "Results Display",
                description: "Professional dashboard presents comprehensive analysis including quality grades, confidence scores, and actionable recommendations."
              }
            ].map((item, index) => (
              <div key={index} style={styles.processBox}>
                <h4 style={{ fontSize: '1.3rem', color: '#2c3e50', marginBottom: '15px', fontWeight: '600' }}>
                  {item.title}
                </h4>
                <p style={{ color: '#666', lineHeight: '1.6' }}>{item.description}</p>
              </div>
            ))}
          </div>
          <div>
            <br />
            <br />
            <br />
            <br />
          </div>
          <button 
            onClick={onNavigateToGrading}
            style={{
              ...styles.ctaButton,
              padding: '20px 50px',
              fontSize: '1.2rem',
            }}
          >
            Start Now
          </button>
          <div>
            <br />
          </div>
        </div>
      </section>

      {/* Start Now Section */}
      <section style={styles.startNowSection}>
        <div style={styles.container}>
          <h2 style={{ fontSize: '2.5rem', marginBottom: '20px', fontWeight: '700' }}>
            Ready to Transform Your Fruit Assessment?
          </h2>
          <p style={{ fontSize: '1.2rem', marginBottom: '40px', opacity: 0.9 }}>
            Experience the power of AI-driven fruit classification and grading
          </p>
        </div>
      </section>
    </div>
  );
};

export default HomePage;
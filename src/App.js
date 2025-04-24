// get wheredoicomefrom.com

import React, { useState, useEffect } from 'react';
import * as ort from 'onnxruntime-web';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [ethnicities, setEthnicities] = useState({
    'East Asian': 0,
    'Indian': 0,
    'Black': 0,
    'White': 0,
    'Middle Eastern': 0,
    'Latino/Hispanic': 0,
    'Southeast Asian': 0
  });
  const [model, setModel] = useState(null);

  useEffect(() => {
    // Load the ONNX model
    const loadModel = async () => {
      try {
        const session = await ort.InferenceSession.create('/ethnicity_prediction_model.onnx');
        setModel(session);
      } catch (error) {
        console.error('Error loading model:', error);
      }
    };
    loadModel();
  }, []);

  const preprocessImage = async (imageFile) => {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = 224;  // Assuming model expects 224x224 images
        canvas.height = 224;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, 224, 224);
        
        // Get image data and normalize to [0, 1]
        const imageData = ctx.getImageData(0, 0, 224, 224);
        const data = new Float32Array(1 * 3 * 224 * 224);
        for (let i = 0; i < imageData.data.length / 4; i++) {
          data[i] = imageData.data[i * 4] / 255.0;     // R
          data[i + 224 * 224] = imageData.data[i * 4 + 1] / 255.0;     // G
          data[i + 2 * 224 * 224] = imageData.data[i * 4 + 2] / 255.0; // B
        }
        resolve(data);
      };
      img.src = URL.createObjectURL(imageFile);
    });
  };

  const softmax = (arr) => {
    const max = Math.max(...arr);
    const exp = arr.map(x => Math.exp(x - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(x => x / sum);
  };

  const predictEthnicity = async (imageData) => {
    if (!model) return;

    try {
      const tensor = new ort.Tensor('float32', imageData, [1, 3, 224, 224]);
      const results = await model.run({ "input.1": tensor });
      const output = Array.from(results['191']['cpuData']);
      const probabilities = softmax(output);
      
      const percentages = {
        'East Asian': Math.round(probabilities[0] * 100),
        'Indian': Math.round(probabilities[1] * 100),
        'Black': Math.round(probabilities[2] * 100),
        'White': Math.round(probabilities[3] * 100),
        'Middle Eastern': Math.round(probabilities[4] * 100),
        'Latino/Hispanic': Math.round(probabilities[5] * 100),
        'Southeast Asian': Math.round(probabilities[6] * 100)
      };
      
      console.log("kl4");
      setEthnicities(percentages);
      console.log("kl5");
    } catch (error) {
      console.error('Error during prediction:', error);
    }
  };

  useEffect(() => {
    const handlePaste = async (e) => {
      const items = e.clipboardData?.items;
      if (!items) return;

      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
          const blob = items[i].getAsFile();
          const url = URL.createObjectURL(blob);
          setImage(url);
          
          // Preprocess image and run prediction
          const imageData = await preprocessImage(blob);
          await predictEthnicity(imageData);
          break;
        }
      }
    };

    document.addEventListener('paste', handlePaste);
    return () => document.removeEventListener('paste', handlePaste);
  }, [model]);

  return (
    <div className="App">
      <div className="container">
        <h1>Guess My Race</h1>
        <div className="content-wrapper">
          <div className="left-panel">
            <div className="image-container">
              {image ? (
                <img src={image} alt="Pasted" className="pasted-image" />
              ) : (
                <div className="placeholder">
                  <p>Press Command+V (or Ctrl+V) to paste an image</p>
                </div>
              )}
            </div>
          </div>
          <div className="right-panel">
            <div className="results">
              <h2>Ethnicity Analysis</h2>
              <div className="ethnicity-bars">
                {Object.entries(ethnicities).map(([ethnicity, percentage]) => (
                  <div key={ethnicity} className="ethnicity-bar">
                    <div className="ethnicity-label">{ethnicity}</div>
                    <div className="bar-container">
                      <div 
                        className="bar" 
                        style={{ width: `${percentage}%` }}
                      >
                        {percentage}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;

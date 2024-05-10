import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import Home from './Home/Home';
import Datasets from './Datasets/Datasets';
import Dataset from './Datasets/Dataset';
import Features from './Features/Features';
import Models from './Models/Models';
import Model from './Models/Model';
import Head from './Head/Head';

import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';

const root = ReactDOM.createRoot(document.getElementById('root'));

root.render(
  <div className="all">
    <Head/>
    <Router>
      <Routes>
        <Route path="/home" element={<Home/>} /> 
        <Route path="/datasets" element={<Datasets/>} /> 
        <Route path="/datasets/sciq" element={<Dataset/>} /> 
        <Route path="/datasets/arceasy" element={<Dataset/>} /> 
        <Route path="/datasets/arcchallenge" element={<Dataset/>} /> 
        <Route path="/datasets/arccombined" element={<Dataset/>} />  
        <Route path="/features" element={<Features/>} /> 
        <Route path="/models" element={<Models/>} /> 
        <Route path="/models/logisticregression" element={<Model/>} /> 
        <Route path="/models/randomforest" element={<Model/>} /> 
        <Route path="/models/lambdamart" element={<Model/>} /> 
        <Route path="/models/neuralnetwork" element={<Model/>} /> 
        <Route path="*" element={<Navigate to="/home" />} />
      </Routes>
    </Router>
  </div>
);

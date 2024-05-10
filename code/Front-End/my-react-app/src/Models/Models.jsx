import React from 'react';
import './Models.css';
import models from './figures/models.png'
import { Link } from "react-router-dom"

import logisticRegression from "./figures/logisticregression.png"
import lambdaMART from "./figures/lambdamart.png"
import randomForest from "./figures/randomforest.png"
import neuralNetwork from "./figures/neuralnetwork.png"

function Bio() {

  let squares = new Array(3)
  squares[0] = new Array(1)
  squares[0][0] = (<div className="model"> <Link to="/models/logisticregression"> <img className="modelStart" src={logisticRegression}/> </Link> </div>)
  squares[1] = new Array(1)
  squares[1][0] = (<div className="model"> <Link to="/models/randomforest"> <img className="modelStart" src={randomForest}/> </Link> </div>)
  squares[2] = new Array(1)
  squares[2][0] = (<div className="model"> <Link to="/models/lambdamart"> <img className="modelStart" src={lambdaMART}/> </Link> </div>)
  squares[3] = new Array(1)
  squares[3][0] = (<div className="model"> <Link to="/models/lambdamart"> <img className="modelStart" src={neuralNetwork}/> </Link> </div>)

  return (
    <div className="Bio2">
      <div className="beforeTitle">
        <br/>
        <img className="title" src={models}/> 
      </div> 
      <div className="afterTitle">
        <div className="Grid">
          
          {squares}
        </div>
      </div>
    </div>
  )
}

function Models() {
  return (
    <div className="Bio">
      <Bio/>
    </div>
  );
}

export default Models;
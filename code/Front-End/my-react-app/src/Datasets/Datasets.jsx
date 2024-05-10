import React from 'react';
import './Datasets.css';
import { Link } from "react-router-dom"
import Popup from 'reactjs-popup'

import datasets from './figures/datasets.png'

import sciq from './figures/sciq.png'
import arcEasy from './figures/arcEasy.png'
import arcChallenge from './figures/arcChallenge.png'
import arcCombined from './figures/arcCombined.png'
import newBook from './figures/newBook.png'


function Contents() {
  let squares = new Array(5)
  squares[0] = new Array(1)
  squares[0][0] = (<div className="model"> <Link to="/datasets/sciq"> <img src={sciq}/> </Link> </div>)
  squares[1] = new Array(1)
  squares[1][0] = (<div className="model"> <Link to="/datasets/arceasy"> <img src={arcEasy}/> </Link> </div>)
  squares[2] = new Array(1)
  squares[2][0] = (<div className="model"> <Link to="/datasets/arcchallenge"> <img src={arcChallenge}/> </Link> </div>)
  squares[3] = new Array(1)
  squares[3][0] = (<div className="model"> <Link to="/datasets/arccombined"> <img src={arcCombined}/> </Link> </div>)
  squares[4] = new Array(1)
  squares[4][0] = (<div className="model"> 
  <Popup trigger={<img src={newBook}/>} modal> 
    <span> 
      <div className="popup">
       <form>
          <label>
            {"Question:"}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            <input type="text" name="question" />
          </label>
          <br/>
          <label>
            {"Correct Answer: "}
            <input type="text" name="answer" />
          </label>
          <input type="submit" value="Submit" />
        </form>
      </div>
    </span>  
  </Popup>
</div>)

  return (
    <div className="Bio2">
      <div className="beforeTitle">
        <br/>
        <img className="title" src={datasets}/> 
      </div> 
      <div className="afterTitle">
      <div className="datasetGrid">
        {squares}
      </div>  
      <br/>
      </div>
    </div> 
  )
}

function Datasets() {
  return (
    <div className="Bio">
      <Contents/>
    </div>
  );
}

export default Datasets;
import React from 'react';
import './Home.css';
import title from './title.png';

function Bio() {
  return (
    <div className="Bio2">
      <br/>
      <img className="title" src={title}/>  
      <div className="afterTitle">
        <p className="Subtitle"> <b> <u> Sections </u></b> </p>
        <p className="Sections"> 
          <b>Datasets:</b> Contains all datasets used in the project, with functionality to add, delete, edit, and remove questions from saved lists.
          <br/>
          <br/>
          <b>Features:</b> List of all features used in feature-based classifiers.
          <br/>
          <br/>
          <b>Models:</b> The feature-based models Logisitic Regression, Random Forest, and LambdaMART, and the neural network-based IRGAN model.
          <br/>
          Go here to train the models, and to recieve a ranked list of answers for any given question. 
        </p>
      </div>
    </div>
  )
}

function Home() {
  return (
    <div className="Bio">
      <Bio/>
    </div>
  );
}

export default Home;
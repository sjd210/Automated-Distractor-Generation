import React from 'react';
import './Datasets.css';
import sciqTrainData from './SciQ dataset-2 3/train.json';
import sciqValidData from './SciQ dataset-2 3/valid.json';
import sciqTestData from './SciQ dataset-2 3/test.json'; 

import arcEasyTrainData from './ARC-Easy-2/train.json';
import arcEasyValidData from './ARC-Easy-2/valid.json';
import arcEasyTestData from './ARC-Easy-2/test.json'; 

import arcChallengeTrainData from './ARC-Challenge-2/train.json';
import arcChallengeValidData from './ARC-Challenge-2/valid.json';
import arcChallengeTestData from './ARC-Challenge-2/test.json';

import arcCombinedTrainData from './ARC-Combined/train.json';
import arcCombinedValidData from './ARC-Combined/valid.json';
import arcCombinedTestData from './ARC-Combined/test.json';

import datasets from './figures/datasets.png'
import paper from './figures/paper.png'
import sciqTrain from './figures/sciqTrain.png'
import sciqValid from './figures/sciqValid.png'
import sciqTest from './figures/sciqTest.png'

import arcEasyTrain from './figures/arcEasyTrain.png'
import arcEasyValid from './figures/arcEasyValid.png'
import arcEasyTest from './figures/arcEasyTest.png' 

import arcChallengeTrain from './figures/arcChallengeTrain.png'
import arcChallengeValid from './figures/arcChallengeValid.png'
import arcChallengeTest from './figures/arcChallengeTest.png'

import arcCombinedTrain from './figures/arcCombinedTrain.png'
import arcCombinedValid from './figures/arcCombinedValid.png'
import arcCombinedTest from './figures/arcCombinedTest.png'

function TableEntry(data, headers) {
  let entries = new Array(headers.length)
  for (let i = 0; i < headers.length; i++) { 
    entries[i] = <td> {data[headers[i]]} </td>
  }

  return(
    <tr>
      {entries}
    </tr>
  )
}

function JsonToTable(data, subtitle, headers) {
  let entries = new Array(data.length)
  for (let i = 0; i < data.length; i++) { 
    entries[i] = TableEntry(data[i], headers)
  } 
  return (
    <div>
      <img className="datasetSubtitle" src={subtitle}/>  
      <div className="tableContainer">
        <div className="paperContainer" style={{backgroundImage: `url(${paper})`, backgroundRepeat: "repeat-y", backgroundSize: "contain"}}>
        </div>
        <div className="tableContainer2">
          <table>
            <tr className="headers">
              <th> Question </th>
              <th> Answer </th>
              <th> Distractor 1</th>
              <th> Distractor 2</th>
              <th> Distractor 3</th>
            </tr>
            {entries}
          </table>
        </div>
      </div>
    </div>
  )
}

function Contents() {
    let url = window.location.href.split("/")[4]
    let train, valid, test, trainData, validData, testData;
    if (url == "arceasy") { train = arcEasyTrain; valid = arcEasyValid; test = arcEasyTest; trainData = arcEasyTrainData; validData = arcEasyValidData; testData = arcEasyTestData; }
    else if (url == "arcchallenge") { train = arcChallengeTrain; valid = arcChallengeValid; test = arcChallengeTest; trainData = arcChallengeTrainData; validData = arcChallengeValidData; testData = arcChallengeTestData; } 
    else if (url == "arccombined") { train = arcCombinedTrain; valid = arcCombinedValid; test = arcCombinedTest; trainData = arcCombinedTrainData; validData = arcCombinedValidData; testData = arcCombinedTestData; } 
    else if (url == "sciq") { train = sciqTrain; valid = sciqValid; test = sciqTest; trainData = sciqTrainData; validData = sciqValidData; testData = sciqTestData; } 

  return (
    <div className="Bio2">
      <br/>
      <img className="title" src={datasets}/>  
      <div className="afterTitle">
      <br/>
      {JsonToTable(trainData, train, ["question", "correct_answer", "distractor1", "distractor2", "distractor3"])}
      <br/>
      {JsonToTable(validData, valid, ["question", "correct_answer", "distractor1", "distractor2", "distractor3"])}
      <br/>
      {JsonToTable(testData, test, ["question", "correct_answer", "distractor1", "distractor2", "distractor3"])}
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
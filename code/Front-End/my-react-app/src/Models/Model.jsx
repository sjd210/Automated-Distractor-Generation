import React, { useState } from 'react';
import './Models.css';
import models from './models.png'
import paper from './paper.png'

import sciqTestData from '../Datasets/SciQ dataset-2 3/test.json'; 
import arcEasyTestData from '../Datasets/ARC-Easy-2/test.json'; 
import arcChallengeTestData from '../Datasets/ARC-Challenge-2/test.json';
import arcCombinedTestData from '../Datasets/ARC-Combined/test.json';

import sciqTestResults from '../Datasets/SciQ dataset-2 3/testResults.json'; 
import arcEasyTestResults from '../Datasets/ARC-Easy-2/testResults.json'; 
import arcChallengeTestResults from '../Datasets/ARC-Challenge-2/testResults.json';
import arcCombinedTestResults from '../Datasets/ARC-Combined/testResults.json';

function ModelInterface(){
  const [dataset, setDataset] = useState("")
  const [question, setQuestion] = useState("")

  function TableEntry(data, headers) {
    let entries = new Array(headers.length)
    for (let i = 0; i < headers.length; i++) { 
      entries[i] = <td> {data[headers[i][1]]} </td>
    }

    return(
      <tr className={"questionOption" + " clicked" + String(question==data[headers[0][1]])} onClick={() => setQuestion(data[headers[0][1]])}>
        {entries}
      </tr>
    )
  }

  function JsonToTable(data, subtitle, headers) {
    let tableHeaders = new Array(headers.length)
    for (let i = 0; i < headers.length; i++) { 
      tableHeaders[i] = <th style={{width: "100%"}}> {headers[i][0]} </th>
    } 
  
    let entries
  
    if (headers[0][0] == "Distractor") {
      let questionVal = data.find(item => item.question == question)
      if (questionVal === undefined) {
        questionVal = {"distractor": new Array(10).fill("undefined")}
      }
  
      entries = new Array(10)
      for (let i = 0; i < 10; i++) { 
        entries[i] = <tr> <td> {questionVal[headers[0][1]][i]} </td> <td/> </tr>
      } 
    } else {
      entries = new Array(data.length)
      for (let i = 0; i < data.length; i++) { 
        entries[i] = TableEntry.bind(this)(data[i], headers)
      } 
    }
    
    return (
      <div>
        <p className="Subtitle"> <b> <u> {subtitle} </u></b> </p>
        <div className="tableContainer">
          <div className="paperContainer" style={{backgroundImage: `url(${paper})`, backgroundRepeat: "repeat-y", backgroundSize: "contain"}}>
          </div>
          <div className="tableContainer2">
            <table>
              <tr className="headers">
                {tableHeaders}
              </tr>
              {entries}
            </table>
          </div>
        </div>
      </div>
    )
  }

  function SelectDataset() {
    return (
      <div>
        <p className="Subtitle"> <b> <u> Select Dataset: </u></b> </p>
        <div className="tableContainer">
          <div className="paperContainer" style={{backgroundImage: `url(${paper})`, backgroundRepeat: "repeat-y", backgroundSize: "contain"}}>
          </div>
          <div className="tableContainer2">
            <table>
              <tr className="headers">
                <th style={{width: "100%"}}> Dataset </th>
              </tr>
              <tr className={"datasetOption" + " clicked" + String(dataset=="SciQ")} onClick={() => setDataset("SciQ")}> <td> SciQ </td> <td/> </tr>
              <tr className={"datasetOption" + " clicked" + String(dataset=="ARC Easy")} onClick={() => setDataset("ARC Easy")}> <td> ARC Easy </td> <td/> </tr>
              <tr className={"datasetOption" + " clicked" + String(dataset=="ARC Challenge")} onClick={() => setDataset("ARC Challenge")}> <td> ARC Challenge </td> <td/> </tr>
              <tr className={"datasetOption" + " clicked" + String(dataset=="ARC Combined")} onClick={() => setDataset("ARC Combined")}> <td> ARC Combined </td> <td/> </tr>
            </table>
          </div>
        </div>
      </div>
    )
  }

  let tables = new Array(2)
  tables[0] = SelectDataset.bind(this)()

  if (dataset=="SciQ") {
    tables[1] = JsonToTable.bind(this)(sciqTestData, "Select Question:", [["Question", "question"], ["Answer", "correct_answer"]])
  } else if (dataset=="ARC Easy") {
    tables[1] = JsonToTable.bind(this)(arcEasyTestData, "Select Question:", [["Question", "question"], ["Answer", "correct_answer"]])
  } else if (dataset=="ARC Challenge") {
    tables[1] = JsonToTable.bind(this)(arcChallengeTestData, "Select Question:", [["Question", "question"], ["Answer", "correct_answer"]])
  } else if (dataset=="ARC Combined") {
    tables[1] = JsonToTable.bind(this)(arcCombinedTestData, "Select Question:", [["Question", "question"], ["Answer", "correct_answer"]])
  }

  if (question != "") {
    tables[3] = JsonToTable.bind(this)(sciqTestResults, "Top 10 Selected Distractors:", [["Distractor", "distractor"]])
  }

  return (
    <div className="Bio">
      <div className="Bio2">
      <div className="beforeTitle">
        <br/>
        <img className="title" src={models}/> 
      </div> 
        <div className="afterTitle">
        <br style={{content: "", margin: "2em", display: "block", "font-size": 5 }}/>
          {tables}
        </div>  
      </div>
    </div>
  );
}

export default ModelInterface;
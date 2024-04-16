import React from 'react';
import './Datasets.css';

import datasets from './figures/datasets.png'
import paper from './figures/paper.png'


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
  return (
    <div className="Bio">
      <br/>
      <img className="title" src={datasets}/>  
      <br/>
      TESt
    </div> 
  )
}

function Datasets() {
  return (
    <div className="aaa">
      <Contents/>
    </div>
  );
}

export default Datasets;
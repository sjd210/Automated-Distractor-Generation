import React from 'react';
import './Models.css';
import models from './models.png'
import { Link } from "react-router-dom"

import paper from './paper.png'

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
        SELECT QUESTION / LIST OF ANSWERS
        <div className="tableContainer">
          <div className="paperContainer" style={{backgroundImage: `url(${paper})`, backgroundRepeat: "repeat-y", backgroundSize: "contain"}}>
          </div>
          <div className="tableContainer2">
            <table>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
                <tr> test </tr>
            </table>
          </div>
        </div>
      </div>
    )
  }

function Models() {
    let a = JsonToTable([{"a": "b"}], {paper}, ["a", "b", ])
  return (
    <div className="Bio">
    <br/>
    <img className="title" src={models}/>  
      {a}
      {a}
    </div>
  );
}

export default Models;
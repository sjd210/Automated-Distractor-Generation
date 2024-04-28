
import React, { useState } from 'react';
import './Datasets.css';
import datasets from './figures/datasets.png'
import paper from './figures/paper.png'


function JsonToTable(data, subtitle, headers) {
    return (
      <div>
        <p className="Subtitle"> <b> <u> {subtitle} </u></b> </p>
        <div className="tableContainer">
          <div className="paperContainer" style={{backgroundImage: `url(${paper})`, backgroundRepeat: "repeat-y", backgroundSize: "contain"}}>
          </div>
          <div className="tableContainer2">
            <table>
              <tr className="headers">
                <th style={{width: "100%"}}> Dataset </th>
              </tr>
              <tr className={"datasetOption" + " clicked" + String(this.state.dataset=="SciQ")} onClick={() => this.setState({dataset: "SciQ"})}> <td> SciQ </td> <td/> </tr>
              <tr className={"datasetOption" + " clicked" + String(this.state.dataset=="ARC Easy")} onClick={() => this.setState({dataset: "ARC Easy"})}> <td> ARC Easy </td> <td/> </tr>
              <tr className={"datasetOption" + " clicked" + String(this.state.dataset=="ARC Challenge")} onClick={() => this.setState({dataset: "ARC Challenge"})}> <td> ARC Challenge </td> <td/> </tr>
              <tr className={"datasetOption" + " clicked" + String(this.state.dataset=="ARC Combined")} onClick={() => this.setState({dataset: "ARC Combined"})}> <td> ARC Combined </td> <td/> </tr>
            </table>
          </div>
        </div>
      </div>
    )
  }

class CustomDatasetInterface extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      dataset: ""
    };

  }

  componentDidMount() {
  }

  render() { 
    let a = JsonToTable.bind(this)([{"a": "b"}], "AAAA", ["question", "correct_answer"])
    return (
      <div className="Bio">
        <div className="Bio2">
          <br/>
          <img className="title" src={datasets}/>
          <div className="afterTitle">
          <br style={{content: "", margin: "2em", display: "block", "font-size": 5 }}/>
            {a}
            {a}
          </div>  
        </div>
      </div>
    );
  }
}

export default CustomDatasetInterface;
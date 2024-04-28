import './App.css';
import React, {useState} from 'react';

import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';
import NavDropdown from 'react-bootstrap/NavDropdown'; 
import {useNavigate} from "react-router-dom";

// ================ HEADER ======================

function setCurrentRobot(robotName) {
  const num = this.state.robotIDs.indexOf(robotName);
  this.setState({ 
    currentRobot : num,
    overview: false,
  })
}

function RobotDropDown() {
  let robList = [];
  let i = 0;

  for (const ID in this.state.robotIDs) {
    robList[i] = <NavDropdown.Item onClick={e => setCurrentRobot.bind(this)(e.target.innerText)}>{this.state.robotIDs[i]}</NavDropdown.Item>
    i++;
  }

  return (
    <NavDropdown title={<span className = {"link navtext navtext" + this.state.colour} >Robots</span>}  >
      {robList}
      <NavDropdown.Divider />
      <NavDropdown.Item onClick={() => this.setState({overview: this.state.loggedIn})}>{this.state.loggedIn ? "Overview" : "Not Logged In"}</NavDropdown.Item>
    </NavDropdown>
  )
}


function ColourDropDown() {
  return (
    <NavDropdown title={<span className = {"navtext link navtext" + this.state.colour}>Themes</span>}>
      <NavDropdown.Item onClick={() => this.state.colour = "White"}>White</NavDropdown.Item>
      <NavDropdown.Item onClick={() => this.state.colour = "Green"}>Green</NavDropdown.Item>
      <NavDropdown.Item onClick={() => this.state.colour = "Dark"}>Dark</NavDropdown.Item>
      <NavDropdown.Item onClick={() => this.state.colour = "Black"}>Black</NavDropdown.Item>
    </NavDropdown>
  )
}

async function HandleLogOut(e, nav){ 
  e.preventDefault();
  let res = await fetch(window.location.origin + "/logout", {
    method: "POST",
    body: ""
  });
  nav("/sign_in");
}

  /* Page structure:
  AutoPickr   Login Button    Robot dropdown            Robot name/Overview              Theme dropdown  Autopickr Icon 
  */

function Header(nav) {
  return (
    <Navbar className={'navbar nav' + this.state.colour} sticky="top">
        <Nav className="brand">
          <Navbar.Brand href="sign_in"> <p className = {"navtext brand navtext" + this.state.colour}>{"⠀AutoPickr"}</p></Navbar.Brand>
          <Nav.Link href="/sign_in"> <p className = {"navtext link navtext" + this.state.colour}> {"Log In"} </p> </Nav.Link>
          {RobotDropDown.bind(this)()}
        </Nav>
        <Navbar.Toggle aria-controls="asic-bnavbar-nav" />
         <Navbar.Collapse id="basic-navbar-nav"> 
          <Nav className="leftnav" >
            
          </Nav>
          <p className="container" style={{display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
              <Navbar.Brand className={"navtext Log" + this.state.loggedIn + this.state.colour}>
                <label className={"navtext Log" + this.state.loggedIn + this.state.colour + " navtext" + this.state.colour}>
                  {this.state.loggedIn ? (this.state.overview ? "Overview" : "Current Robot: " + this.state.robotIDs[this.state.currentRobot]) : "⠀Not logged in⠀"}
                </label>
              </Navbar.Brand>         
          </p>
          <Nav className="ms-auto">
            {ColourDropDown.bind(this)()} 
            <Nav.Link onClick={e => HandleLogOut(e,nav)}> <p className={"navtext link navtext" + this.state.colour}>Log Out</p></Nav.Link>
            <img src={"autopickr" + this.state.colour + ".png"} style={{ width: 50, height: 50 }}/>
          </Nav>
        </Navbar.Collapse> 
    </Navbar>
  );
}

// ================ FOOTER ======================

  /* Page structure:
  Time            Current State             Autopickr Icon 
  */

function Footer() {
  let time = new Date();
  return (
    <Navbar className={'footer nav' + this.state.colour} sticky="bottom">
      <Navbar.Brand style={{width: 0}}>
        <p className = {"navtext navtext" + this.state.colour}>
          {"⠀" + time.getHours() + ":" + (((time.getMinutes()) < 10) ? "0" : "") + time.getMinutes() + ":" + (((time.getSeconds()) < 10) ? "0" : "") + time.getSeconds()}
        </p>
      </Navbar.Brand>
      <Navbar.Collapse id="basic-navbar-nav"> 
        <p className="container" style={{display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
          <Nav className = {"navtext navtext" + this.state.colour}>{"⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀CURRENT STATUS: " + ((this.state.currentRobot !== null) ? this.state.robots[this.state.currentRobot].last_reading : "")}</Nav>
        </p>
        <Navbar.Brand className="ms-auto">
          <img src={"autopickr" + this.state.colour + ".png"} style={{ width: 50, height: 50 }}/>
        </Navbar.Brand>
      </Navbar.Collapse> 
    </Navbar>
  );
}

// ================ STATE SQUARES ======================

function Square(num) {
  let a = "";
  if (this.state.currentRobot !== null) {
    if ((num == 3) && (this.state.squares[3].active)) {
      a = this.state.robots[this.state.currentRobot].parameter + "cm away"
    }
    else if ((num == 4) && (this.state.squares[4].active)) {
      a = this.state.robots[this.state.currentRobot].parameter + " mag strength";
    }
  }
  
  let time = new Date(this.state.squares[num].timestamp);

   /* Page strcture:
  Status icon
  Status name
  Additonal robot data
  Time in state
  */

  return (
    <div className = "button-container">
      <button className={"square button" + this.state.colour + " " + (this.state.squares[num].active ? this.state.squares[num].stateName + " " + this.state.squares[num].stateName + this.state.colour : "")} >
        <div className="button-icon" >
          <img src={this.state.squares[num].stateName + ((this.state.squares[num].active && this.state.colour !== "Black") ? "-Colour.png" : ".png")} width="75%" />
        </div>
          <p>
            <br />
            <p style={{"font-size": "3.5vh"}}>
              {this.state.squares[num].stateName}
            </p>
            <br />
            <br />
            {this.state.squares[num].active ? ("Since " + time.getHours() + ":" + (((time.getMinutes()) < 10) ? "0" : "") + time.getMinutes() + ":" + (((time.getSeconds()) < 10) ? "0" : "") + time.getSeconds()) : ""}
            <br />
            {a}
          </p>
      </button>
      <img className="button-background" src="ButtonTransp.png"/>
    </div>
  );
}

function Grid(i) {
  let squares = Array(i);
  for (let j = 0; j < i; j++) {
    squares[j] = Square.bind(this)(j);
  }
  // All states shown on screen
  return (
    <div className="grid-row">
      {squares}
    </div>
  )
}

// ================ OVERVIEW PAGE ====================== 

function RobotButton(i) {
  let a = "";
  if (this.state.currentRobot !== null) {
    if (this.state.robots[i].last_reading == "Lifeform-Detected") {
      a = this.state.robots[i].parameter + "cm away"
    }
    else if (this.state.robots[i].last_reading == "Blade-Unseated") { 
      a = this.state.robots[i].parameter + " mag strength";
    }
  }

  const time = new Date(this.state.robots[i].timestamp_state_start);

  /* Page strcture:
  Robot name       Robot Status + Icon / Additional Robot Information      Time in State
  */

  return (
    <div className={"robotSquareContainer "}>
    <Navbar className={"robotSquare " + this.state.robots[i].last_reading + " " + this.state.robots[i].last_reading + this.state.colour}>
        <Nav className="brand">
          <Navbar.Brand className = "navtext robotID" style={{"font-size": "10vh"}}>{"⠀" + this.state.robotIDs[i]}</Navbar.Brand>
        </Nav>
        <Navbar.Toggle aria-controls="asic-bnavbar-nav" />
         <Navbar.Collapse id="basic-navbar-nav"> 
          <p className="container" style={{display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
              <Navbar.Brand className = "navtext" style={{"font-size": "5vh"}}>{("Current Status: " + this.state.robots[i].last_reading)}</Navbar.Brand>
              <img src= {this.state.robots[i].last_reading + (this.state.colour == "Black" ? ".png" : "-Colour.png")} style={{"width": "15vh"}} /> 
              <p style={{"font-size": "2.5vh"}}>
                {a} 
              </p>    
          </p>
          <Nav className="ms-auto">
            <Nav.Link style={{ textDecoration: 'underline', "font-size": "2.5vh"}} className = "navtext">{("Since " + time.getHours() + ":" + (((time.getMinutes()) < 10) ? "0" : "") + time.getMinutes() + ":" + (((time.getSeconds()) < 10) ? "0" : "") + time.getSeconds())}</Nav.Link>
          </Nav>
        </Navbar.Collapse> 
    </Navbar>
    </div>
  );
}

function RobotGrid() {
  let robotButtons = Array(this.state.robotIDs.length);
  for (let i = 0; i < this.state.robotIDs.length; i++) {
    robotButtons[i] = RobotButton.bind(this)(i);    // Displays all robots in an overview
  }

  return (
    <div className ="robotGrid">
      {robotButtons} 
    </div>
  )
}

// ================ ENTIRE PAGE ====================== 

function fetchRobot(id, num) { // Called whenever fetching data for a given robot
  let stateNo = -1;
  fetch(window.location.origin + "/api/robot/" + id).then((response) => response.json()).then(function(input_data) {
    this.state.robots[num] = input_data;

    switch (input_data.last_reading) {  // Converts from database format to internal state format
      case "Moving":
        this.state.robots[num].last_reading = "Moving"; // 0
        stateNo = 0;
        break;
      case "StopHarvesting":
        this.state.robots[num].last_reading = "Harvesting"; // 1
        stateNo = 1;
        break;
      case "StopRouteEnded":
        this.state.robots[num].last_reading = "Route-Ended";  // 2
        stateNo = 2;
        break;
      default:
        if ("StopLifeformDetected" in input_data.last_reading) {
          this.state.robots[num].parameter = input_data.last_reading.StopLifeformDetected;
          this.state.robots[num].last_reading = "Lifeform-Detected";  // 3
          stateNo = 3;
        }
        else if ("StopBladeUnseated" in input_data.last_reading) {
          this.state.robots[num].parameter = input_data.last_reading.StopBladeUnseated;
          this.state.robots[num].last_reading = "Blade-Unseated"; // 4
          stateNo = 4;
        }
        else {
          this.state.robots[num].last_reading = "Network-Error";
        }
        break;
    }

    if (input_data.id == this.state.robotIDs[this.state.currentRobot]) { // Moves data from robot state to squares state
      for (let i = 0; i < this.state.squares.length; i++) {
        this.state.squares[i].active = false;
      }
      this.state.squares[stateNo].active = true;
      this.state.squares[stateNo].timestamp = input_data.timestamp_state_start;
    }

    if (this.state.currentRobot == null && this.state.robots[0] !== null) { // Sets default robot to robot 0
      setCurrentRobot.bind(this)(this.state.robots[0].id)
    }

    this.setState({
      robots : this.state.robots,
      loggedIn : true,
      squares : this.state.squares,
    })

    /* if (num == 0) {
      console.log(this.state);
    } */
  }.bind(this));
}

class FarmerInterface extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      squares: new Array(5).fill(null).map(function() { return (
        {
        stateName: "none",
        active: false,
        timestamp: Date.now(),
        }) }),  // 5 MAIN STATE SQUARES 

      robotIDs: {},
      robots: {},
      currentRobot: null,
      noInterval: true,
      overview: false,
      loggedIn: false,
      colour: "Green"
    };

  }

  componentDidMount() {
    this.state.squares[0].stateName = "Moving";             // Setting all reachable robot states
    this.state.squares[1].stateName = "Harvesting";
    this.state.squares[2].stateName = "Route-Ended";
    this.state.squares[3].stateName = "Lifeform-Detected";
    this.state.squares[4].stateName = "Blade-Unseated";

    this.setState({
      squares: this.state.squares
    }) 

    fetch(window.location.origin + "/api/robot_ids").then((response) => response.json()).then(function(ids) { // Creating an initial list of robots for the given account
      this.setState({
        robotIDs: ids
      });
      for (var id of ids) {
        fetchRobot.bind(this)(id);
      }
    }.bind(this));

    if (this.state.noInterval) {  // Polling every 5 seconds
      this.setState({ noInterval: false })
      let interval = setInterval(() => {
        console.log("calling all robots");
        for (let i = 0; i < this.state.robotIDs.length; i++) {
          fetchRobot.bind(this)(this.state.robotIDs[i], i);
        }
        this.setState({
          robots: this.state.robots
        })
      }, 500);
    }
  } 

  render() {  // Renders either a STATES or OVERVIEW screen
    if (!this.state.overview) {
    return (
      <div className={"STATES back" + this.state.colour}>
        {Header.bind(this)(this.props.nav)} 
        {Grid.bind(this)(5)}
        {Footer.bind(this)()}
    </div> 
    );
    }

    else {
      return (
        <div className={"OVERVIEW back" + this.state.colour}>
          {Header.bind(this)(this.props.nav)} 
          {RobotGrid.bind(this)()}
        </div> 
      );
    }
  }
}

function FarmerInterfaceFun() { // Ambiguates to functional form to allow useNavigate hook to be passed in
  const navigate = useNavigate();

  return (
  <FarmerInterface
    nav={navigate}      
  />);
}

// ========================================

export default FarmerInterfaceFun; 
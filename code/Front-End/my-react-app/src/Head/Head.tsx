import React from 'react';
import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';
import './Head.css';
import icon from './icon.png'

function Banner() {
    return (
        <div className="Banner" />
    )
}
  
function Header() {
    let active = window.location.pathname.split('/')[1]

    return (
        <div className="Nav"> 
            <img className="icon" src={icon}/>
            <Navbar className="Navs" sticky="top" style={{display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
                <Nav.Link href="/home"> <p className={String(active=="home")}> {"Home"} </p> </Nav.Link>
                <p> ⠀ </p>
                <Nav.Link href="/datasets"> <p className={String(active=="datasets")}> {"Datasets"} </p> </Nav.Link>
                <p> ⠀ </p>
                <Nav.Link href="/features"> <p className={String(active=="features")}> {"Features"} </p> </Nav.Link>
                <p> ⠀ </p>
                <Nav.Link href="/models"> <p className={String(active=="models")}> {"Models"} </p> </Nav.Link>
            </Navbar>
        </div>
    )
} 

function Head() {
    return(
        <div className="container">
            <Header/>
        </div>
    );
}

export default Head;
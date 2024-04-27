import React, { useContext, useRef } from 'react';
import './Navbar.css';
import { Link } from 'react-router-dom';
import logo from '../Assets/logo.png';
import cart_icon from '../Assets/cart_icon.png';
import { ShopContext } from '../../Context/ShopContext';

const Navbar = () => {
  const { getTotalCartItems } = useContext(ShopContext);
  const searchInputRef = useRef(); 

  const handleSearch = () => {
    const searchTerm = searchInputRef.current.value.trim(); 
    if (searchTerm) {
      window.location.href = `/search?query=${encodeURIComponent(searchTerm)}`;
    } else {
      console.log("Search term is empty");
    }
  }

  return (
    <div className='nav'>
      <Link to='/' style={{ textDecoration: 'none' }} className="nav-logo">
        <img src={logo} alt="logo" />
        <p>Fashion Recommendation System</p>
      </Link>
      <div className="nav-search">
        <input ref={searchInputRef} type="text" placeholder="Search" />
        <button onClick={handleSearch}>Search</button>
      </div>
      <div className="nav-login-cart">
        {localStorage.getItem('auth-token')
          ? <button onClick={() => { localStorage.removeItem('auth-token'); window.location.replace("/"); }}>Logout</button>
          : <Link to='/login' style={{ textDecoration: 'none' }}><button>Login</button></Link>}
        <Link to="/cart"><img src={cart_icon} alt="cart" /></Link>
        <div className="nav-cart-count">{getTotalCartItems()}</div>
      </div>
    </div>
  )
}

export default Navbar;

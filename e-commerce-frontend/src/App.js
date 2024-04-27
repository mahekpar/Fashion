import Navbar from "./Components/Navbar/Navbar";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Cart from "./Pages/Cart";
import Product from "./Pages/Product";

import LoginSignup from "./Pages/LoginSignup";
import SearchResultsPage from "./Pages/SearchResultsPage"

function App() {

  return (
    <div>
      <Router>
        <Navbar />
        <Routes>
        
          <Route path='/product' element={<Product />}>
            <Route path=':productId' element={<Product />} />

          </Route>
          <Route path="/cart" element={<Cart />} />
          <Route path="/login" element={<LoginSignup/>} />
          <Route path="/search" element={<SearchResultsPage />} />

        </Routes>
        
      </Router>
    </div>
  );
}

export default App;

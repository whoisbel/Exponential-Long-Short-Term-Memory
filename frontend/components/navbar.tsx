import React from "react";

const Navbar = () => {
  return (
    <div className="flex w-full bg-neutral-300 ">
      <ul className="flex ml-1 mt-2 ">
        <li className="flex bg-white rounded-t-md text-center font-light cursor-pointer">
          <a href="/" className="px-10 py-2">
            Something
          </a>
        </li>
        <li className="bg-neutral-200 rounded-t-md flex shadow-inner shadow-neutral-500 font-light cursor-pointer">
          <a href="/prediction" className="px-10 py-2">
            Prediction
          </a>
        </li>
      </ul>
    </div>
  );
};

export default Navbar;

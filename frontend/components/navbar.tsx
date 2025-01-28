"use client";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
const Navbar = () => {
  const pathName = usePathname();
  console.log(pathName);
  const inactiveClass =
    "bg-neutral-200 rounded-t-md flex shadow-inner shadow-neutral-500 font-light cursor-pointer";

  return (
    <div className="flex w-full bg-neutral-300 ">
      <ul className="flex ml-1 mt-2 ">
        <li
          className={`flex  rounded-t-md text-center font-light cursor-pointer ${
            pathName == "/" ? "bg-white" : inactiveClass
          }`}
        >
          <a href="/" className="px-10 py-2">
            Something
          </a>
        </li>
        <li
          className={`flex  rounded-t-md text-center font-light cursor-pointer ${
            pathName == "/prediction" ? "bg-white" : inactiveClass
          }`}
        >
          <a href="/prediction" className="px-10 py-2">
            Prediction
          </a>
        </li>
      </ul>
    </div>
  );
};

export default Navbar;

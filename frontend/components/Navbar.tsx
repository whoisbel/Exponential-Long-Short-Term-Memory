import Link from "next/link";

const Navbar = () => {
  return (
    <div className="bg-gray-900 text-white py-4 px-6 sticky top-0 z-40">
      <div className="container mx-auto flex items-center ">
        <Link href="/" className="text-xl font-bold">
          L'Air Liquide S.A Prediction
        </Link>
      </div>
    </div>
  );
};

export default Navbar;

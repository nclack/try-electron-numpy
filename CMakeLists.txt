cmake_minimum_required(VERSION 3.0)

project(try-electron)

# NOTE: This is probably not running inside a virtual environment even if you
#       have one set up. It's going to use the system default python3 and you'll 
#       need to install NumPy there.
find_package(Python3 REQUIRED Development.Embed NumPy)

add_subdirectory(iffy)
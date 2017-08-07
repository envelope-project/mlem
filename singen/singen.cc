#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <stdint.h>

int main(int argc, char** argv) {
  uint32_t length;

  if(argc != 2) {
    std::cout << "Please specify only lengths!" << std::endl;
    return 0;
  }

  length = std::strtoul(argv[1],nullptr,0);
  std::cout << "Requested length: " << length << std::endl;

  std::ostringstream buffer;
  buffer << "sino" << length << ".sin";

  std::ofstream stream(buffer.str(), std::ios::trunc | std::ios::binary | std::ios::out); 
  if(!stream.is_open()) {
    std::cout << "Unable to open file!" << std::endl;
    return 0;
  }

  int* sino = new int[length]();

  int tmp;
  for(size_t i=0; i<length; ++i) {
    tmp = std::rand() % 50;
    tmp -= 30;
    tmp = (tmp >= 0) ? tmp : 0;
    sino[i] = tmp;
  }

  stream.write(reinterpret_cast<char*>(sino),length*sizeof(int));
  stream.close();

  delete[] sino;

  return 0;
}

/**
    Copyright © 2017 Thorsten Fuchs
    Copyright © 2017 LRR, TU Muenchen
    Authors: Thorsten Fuchs

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
    OTHER DEALINGS IN THE SOFTWARE.

    The above copyright notice and this permission notice shall be 
    included in all copies and/or substantial portions of the Software,
    including binaries.
*/


#include <stdlib.h>
#include <stdint.h>
#include <stdexcept>
#include <iostream>
using std::cout;
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "csr4.h"
#include "csr4generatorRand.h"
#include "csr4storer.h"
#include "csr4loader.h"
#include "configloader.h"

typedef struct Args {
  std::string config;
  float density;
  float variance;
  std::string filename;
} Args;

int processArgs(Args& args, int argc, char** argv) {
  try {
    po::options_description desc("Options");
    desc.add_options()
      ("help,h",
       "Print help messages")
      ("config,c", po::value<std::string>(),
       "Scanner config file")
      ("density,d", po::value<float>(),
       "Ratio of non zero elements (0.0 - 1.0)")
      ("variance,v", po::value<float>(),
       "Variance around mean number of values per row")
      ("filename,f", po::value<std::string>(),
       "Filename of the resulting matrix")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if(vm.count("help")) {
      cout << desc << "\n";
      return 1;
    }

    if(vm.count("config")) {
      args.config = vm["config"].as<std::string>();
    } else {
      cout << desc << "\n";
      return 1;
    }
    if(vm.count("density")) {
      args.density = vm["density"].as<float>();
    } else {
      cout << desc << "\n";
      return 1;
    }
    if(vm.count("variance")) {
      args.variance = vm["variance"].as<float>();
    } else {
      cout << desc << "\n";
      return 1;
    }
    if(vm.count("filename")) {
      args.filename = vm["filename"].as<std::string>();
    } else {
      cout << desc << "\n";
      return 1;
    }
  } catch (const po::error& e) {
    cout << e.what() << std::endl;
    return 2;
  }

  return 0;
}

int main(int argc, char** argv) {
  int ret;
  Args args;
  CSR::CSR4Storer<float> storer;
  CSR::CSR4Loader<float> loader;

  ret = processArgs(args, argc, argv);
  if(ret!=0) {
    return EXIT_SUCCESS;
  }

  cout << "Config: " << args.config << std::endl;
  cout << "Density: " << args.density << std::endl;
  cout << "Variance: " << args.variance << std::endl;
  cout << "Filename: " << args.filename << std::endl;

  CSR::CSR4<float> matrix;
  CSR::CsrResult res;
  res = loader.open(args.filename);
  if(res != CSR::CSR_SUCCESS) {
    std::cerr << "Error loading matrix!" << std::endl;
    return EXIT_FAILURE;
  }
  res = loader.load(matrix);

  storer.open(args.filename + "out");
  storer.save(matrix);

  return EXIT_SUCCESS;
}

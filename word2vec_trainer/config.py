import configparser
import logging


def validate():
    error_msg = ""
    newline = "\n => "

    section = "GLOVE"
    parameter = "filepath"
    ext = (".txt")
    if(not CONFIG[section][parameter].endswith(tuple(ext))):
        error_msg = error_msg + newline + section + "/" + parameter + " MUST be one of " +  repr(ext) + " files."

    if error_msg != "":
        logging.warn( "Config file has following errors: ", error_msg)
        exit(1)



#program = os.path.basename(sys.argv[0])
# TODO: HARDCODED name here
logger = logging.getLogger("IDE")

#EVERTHING: function name/module name/path to source file
logging.basicConfig(format='%(asctime)s:%(levelname)s: %(module)s:%(funcName)s :  %(message)s: %(name)s : %(pathname)s')
logging.root.setLevel(level=logging.INFO)

logger.info("Loading Intent Discovery Engine Configuration...")
#################################
CONFIG = configparser.SafeConfigParser()
CONFIG.read("ideconf.ini")
validate()
##################################
logger.info("Configuration loaded and validated.")

        


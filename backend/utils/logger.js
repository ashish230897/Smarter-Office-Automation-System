/**
 * http://usejsdoc.org/
 */
var winston = require('winston');
winston.emitErrs = true;

var logger = new winston.Logger({
	transports : [ 
  new winston.transports.DailyRotateFile({
	  	level : 'info',
	  	name : 'info-file',
		datePattern : '.yyyy-MM-dd',
		filename : './logs/info-log_file.log',
		colorize : true
	}),
	new winston.transports.DailyRotateFile({
	  	level : 'debug',
	  	name : 'debug-file',
		datePattern : '.yyyy-MM-dd',
		filename : './logs/debug-log_file.log',
		colorize : true
	}),
	new winston.transports.DailyRotateFile({
	  	level : 'error',
	  	name : 'error-file',
		datePattern : '.yyyy-MM-dd',
		filename : './logs/error-log_file.log',
		colorize : true
	}),
new winston.transports.Console({
		level : 'info',
		handleExceptions : true,
		colorize : true
	})

	],
	exitOnError : false
});

module.exports = logger;
module.exports.stream = {
	write : function(message, encoding) {
		logger.info(message);
	}
};
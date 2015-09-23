"""Module that defines the Options class.
"""
class Options(object):
    """Class made to handle models' options.

    This class handles options and do validation checks on them. Normaly an
    instance of this class will be given by a model's get_options method, since
    the model is responsable for setting the validation settings of the options.

    """

    def __init__(self):
        """Initializes the Options instance with no options set

        """
        self.__ops = {}

    #pylint: disable-msg=too-many-arguments
    def add(self, name, value=None,
            required=False, readonly=False,
            value_type=object, description=None):
        """Adds an option.

        Args:
            name (str): The name of the option.
            value (Optional): The value of the option. Set it if a default
                value is wanted. The value's type should follow the value_type
                argument.
            required (Optional[bool]): Sets if this option is required. This
                constraint will be checked in the check method. Default is False
            readonly (Optional[bool]): Sets if this option is read-only. If
                True, every attempt to modify it with the set method will raise
                an AttributeError.
            value_type (Optional[type|Iterable[type]]): If set to a type, the
                option's value will be checked (at the check method) with the
                isintance function. If set to an iterable of types, the
                isinstance function will be used for all types specified and the
                check will succeed if any of the isinstance function calls
                returns True.
            description (Optional[str]): String containing a small description
                of what this option does.

        """
        if name in self.__ops:
            raise ValueError("Option {0} already exist.".format(name))
        new_op = {
            'value': value,
            'required': required,
            'readonly': readonly,
            'value_type': value_type,
            'description': description
        }
        self.__ops[name] = new_op

    def set(self, name, value):
        """Sets an option value.

        Note:
            If an option with the specified name doesn't exist yet, then one is
            created without any optional arguments.
            To add an option, the add method is preferable.

        Args:
            name (str): The name of the option to be set.
            value: The value of the option. This value should follow the
                specified type(s), or else the check method will fail. To see
                the specified type(s), do options.get_all(name)['value_type'].

        """
        if name not in self.__ops:
            self.add(name, value)
            return

        option = self.__ops[name]

        if option['readonly']:
            raise AttributeError("Option {0} is read-only.".format(name))

        if isinstance(option['value_type'], list):
            checked = False
            for value_type in option['value_type']:
                if isinstance(value, value_type):
                    checked = True
                    break
            if not checked:
                raise ValueError("Option {0} should be one of the types:" + \
                                " {1}".format(name, option['value_type']))
        else:
            if not isinstance(value, option['value_type']):
                raise ValueError("Option {0} should be of type {1}.".format(
                                    name, option['value_type']
                                ))

        self.__ops[name]['value'] = value

    def get(self, name):
        """Gets an option's value

        Args:
            name (str): The options name.

        Returns:
            The option's value

        """

        #Just let the KeyError propagate if the option doesn't exist
        return self.__ops[name]['value']

    def get_all(self, name):
        """Gets all the of the option's settings

        Args:
            name (str): The option's name.

        Returns:
            All of the option's settings.

        """

        #Just let the KeyError propagate if the option doesn't exist
        return self.__ops[name]

    def check(self):
        """Runs all the validation checks specified in the options.
        If the check fails, a ValueError will be raised.
        """
        for name in self.__ops:
            option = self.__ops[name]
            if option['required'] and option['value'] is None:
                raise ValueError("Option {0} is required.".format(name))

    def set_from_dict(self, dictionary):
        """Sets all name->value pairs in a dictionary to the options.
        
        The key of each entry in the dictionary will be considered an option's
        name and the value of the entry it's value.

        Args:
            dictionary (dict): Dictionary to be converted to options.

        """
        for name in dictionary:
            self.set(name, dictionary[name])

    def show_all(self):
        """Prints all options.
        """
        import pprint
        pprint.pprint(self.__ops)


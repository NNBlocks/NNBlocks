# NNBlocks is a Deep Learning framework for computational linguistics.
#
#   Copyright (C) 2015 Frederico Tommasi Caroli
#
#   NNBlocks is free software: you can redistribute it and/or modify it under
#   the terms of the GNU General Public License as published by the Free
#   Software Foundation, either version 3 of the License, or (at your option)
#   any later version.
#
#   NNBlocks is distributed in the hope that it will be useful, but WITHOUT ANY
#   WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#   FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
#   details.
#
#   You should have received a copy of the GNU General Public License along with
#   NNBlocks. If not, see http://www.gnu.org/licenses/.

"""Module that defines the Options class.
"""
class Options(object):
    """Class made to handle models' options.

    This class handles options and do validation checks on them. Normaly an
    instance of this class will be given by a model's init_options method, since
    the model is responsable for setting the validation settings of the options.

    """

    def __init__(self):
        """Initializes the Options instance with no options set
        """
        self.__ops = {}

    #pylint: disable-msg=too-many-arguments
    def add(self, name, value=None,
            required=False, readonly=False,
            value_type=object):
        """Adds an option.

        Args:
            name (str): The name of the option.
            value (Optional): The value of the option. Set it if a default
                value is wanted. The value's type should follow the value_type
                argument, but this won't be checked here or anywhere else.
            required (Optional[bool]): Sets if this option is required. This
                constraint will be checked in the check method. Default is False
            readonly (Optional[bool]): Sets if this option is read-only. If
                True, every attempt to modify it with the set method will raise
                an AttributeError.
            value_type (Optional[type|Iterable[type]]): If set to a type, the
                option's value will be checked with the isintance function. If
                set to an iterable of types, the isinstance function will be
                used for all types specified and the check will succeed if any
                of the isinstance function calls returns True.

        """
        if name in self.__ops:
            raise ValueError("Option {0} already exist.".format(name))
        new_op = {
            'value': value,
            'required': required,
            'readonly': readonly,
            'value_type': value_type,
        }
        self.__ops[name] = new_op

    def set(self, name, value):
        """Sets an option value.

        Note:
            If an option with the specified name doesn't exist yet, then one is
            created without any optional settings. A warning will be issued.
            To add an option, the add method is preferable.

        Args:
            name (str): The name of the option to be set.
            value: The value of the option. This value should follow the
                specified type(s), or else this method will fail. To see the
                specified type(s), do options.get_all(name)['value_type'].

        """
        if name not in self.__ops:
            import warnings
            message = "Trying to set a nonexistent option {0}. The option " + \
                        "will be added."
            message = message.format(name)
            warnings.warn(message, RuntimeWarning)
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
                raise ValueError("Option {0} should be of one of the types:" + \
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


/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: core-metadata
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.dao;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import org.edgexfoundry.domain.meta.Asset;
import org.edgexfoundry.domain.meta.Command;
import org.edgexfoundry.domain.meta.Device;
import org.edgexfoundry.domain.meta.DeviceProfile;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/**
 * Convience utility class around DeviceProfile repository
 * 
 */
@Component
public class DeviceProfileDao {

  @Autowired
  private DeviceProfileRepository repos;

  @Autowired
  private DeviceRepository deviceRepos;

  public DeviceProfile getById(String id) {
    return repos.findOne(id);
  }

  public DeviceProfile getByName(String name) {
    return repos.findByName(name);
  }

  public DeviceProfile getByIdOrName(DeviceProfile profile) {
    if (profile == null)
      return null;
    if (profile.getId() != null)
      return repos.findOne(profile.getId());
    if (profile.getName() != null)
      return repos.findByName(profile.getName());
    return null;
  }

  public void checkCommandNames(List<Command> commands) {
    // No two commands for a given profile can have the same name. Command
    // names are not unique across all of EdgeX, but command names per
    // profile must be unique. This method checks the names and throws a
    // DataValidationError if they are determined not to be unique. Should
    // be called from any add or update method.
    if (commands == null || commands.isEmpty())
      return;
    Set<String> uniqueCommandNames =
        commands.stream().map(Command::getName).distinct().collect(Collectors.toSet());
    if (uniqueCommandNames.size() < commands.size())
      throw new DataValidationException("Command names must be unique per DeviceProfile");
  }

  public void checkCommandNames(List<Command> commands, String newCmdName) {
    // No two commands for a given profile can have the same name. Command
    // names are not unique across all of EdgeX, but command names per
    // profile must be unique. This method checks the names and throws a
    // DataValidationError if they are determined not to be unique. Should
    // be called from any add or update method.
    if (commands == null || commands.isEmpty() || newCmdName == null || newCmdName.isEmpty())
      return;
    if (commands.stream().map(Command::getName).anyMatch(newCmdName::equals))
      throw new DataValidationException("Command names must be unique per DeviceProfile");
  }


  public List<DeviceProfile> getAssociatedProfilesForCommand(Command command) {
    if (command == null) {
      return new ArrayList<>();
    }
    return repos
        .findAll().stream().filter(p -> p.getCommands() != null && p.getCommands().stream()
            .map(Command::getId).collect(Collectors.toSet()).contains(command.getId()))
        .collect(Collectors.toList());
  }

  public List<Asset> getOwningServices(DeviceProfile profile) {
    return getAssociatedDevices(profile).stream().map(Device::getService)
        .collect(Collectors.toList());
  }

  private List<Device> getAssociatedDevices(DeviceProfile profile) {
    return deviceRepos.findByProfile(profile);
  }
}

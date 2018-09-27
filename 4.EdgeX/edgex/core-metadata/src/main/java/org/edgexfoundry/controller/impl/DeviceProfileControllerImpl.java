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

package org.edgexfoundry.controller.impl;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import org.edgexfoundry.controller.Action;
import org.edgexfoundry.controller.DeviceProfileController;
import org.edgexfoundry.dao.CommandRepository;
import org.edgexfoundry.dao.DeviceProfileDao;
import org.edgexfoundry.dao.DeviceProfileRepository;
import org.edgexfoundry.dao.DeviceRepository;
import org.edgexfoundry.dao.ProvisionWatcherRepository;
import org.edgexfoundry.domain.meta.ActionType;
import org.edgexfoundry.domain.meta.Command;
import org.edgexfoundry.domain.meta.Device;
import org.edgexfoundry.domain.meta.DeviceProfile;
import org.edgexfoundry.exception.controller.ClientException;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.data.domain.Sort;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.yaml.snakeyaml.DumperOptions.FlowStyle;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.nodes.Tag;

@RestController
@RequestMapping("/api/v1/deviceprofile")
public class DeviceProfileControllerImpl implements DeviceProfileController {

  private static final org.edgexfoundry.support.logging.client.EdgeXLogger logger =
      org.edgexfoundry.support.logging.client.EdgeXLoggerFactory
          .getEdgeXLogger(DeviceProfileControllerImpl.class);

  private static final String ERR_GET = "Error getting profiles:  ";

  @Value("${read.max.limit}")
  private int maxLimit;

  @Autowired
  private DeviceProfileRepository repos;

  @Autowired
  private CommandRepository commandRepos;

  @Autowired
  private DeviceProfileDao dao;

  @Autowired
  private DeviceRepository deviceRepos;

  @Autowired
  private ProvisionWatcherRepository watcherRepos;

  @Autowired
  private CallbackExecutor callback;

  /**
   * Fetch a specific profile by database generated id. May return null if no profile with the id is
   * found. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if no profile can not be found by id.
   * 
   * @param String database generated id for the profile
   * 
   * @return device profile matching on id
   */
  @RequestMapping(method = RequestMethod.GET, value = "/{id}")
  @Override
  public DeviceProfile deviceProfile(@PathVariable String id) {
    try {
      DeviceProfile profile = repos.findOne(id);
      if (profile == null)
        throw new NotFoundException(DeviceProfile.class.toString(), id);
      return profile;
    } catch (NotFoundException nfE) {
      throw nfE;
    } catch (Exception e) {
      logger.error("Error getting profile:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Fetch the profile identified by database generated id and return as a YAML string. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if no profile can not be found by id.
   * 
   * @param String database generated id for the profile
   * 
   * @return device profile in YAML format
   */
  @RequestMapping(method = RequestMethod.GET, value = "/yaml/{id}")
  @Override
  public String deviceProfileAsYaml(@PathVariable String id) {
    try {
      DeviceProfile profile = repos.findOne(id);
      if (profile != null)
        return new Yaml().dumpAs(profile, Tag.MAP, FlowStyle.AUTO);
      throw new NotFoundException(DeviceProfile.class.toString(), id);
    } catch (NotFoundException nfE) {
      throw nfE;
    } catch (Exception e) {
      logger.error("Error getting profile:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return the DeviceProfile matching given name (profile names should be unique). May be null if
   * no profiles matches on the name provided. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns NotFoundException (HTTP 404) if no profile can not be found by
   * name.
   * 
   * @param name
   * @return device profile matching on name
   */
  @RequestMapping(value = "/name/{name:.+}", method = RequestMethod.GET)
  @Override
  public DeviceProfile deviceProfileForName(@PathVariable String name) {
    try {
      DeviceProfile profile = dao.getByName(name);
      if (profile == null)
        throw new NotFoundException(DeviceProfile.class.toString(), name);
      return profile;
    } catch (NotFoundException nfE) {
      throw nfE;
    } catch (Exception e) {
      logger.error("Error getting device profiles:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return, in yaml form, the DeviceProfiles matching given name (profile names should be unique).
   * May be null if no profiles matches on the name provided. Returns ServiceException (HTTP 503)
   * for unknown or unanticipated issues.
   * 
   * @param name
   * @return device profile in YAML matching on name
   */
  @RequestMapping(value = "/yaml/name/{name:.+}", method = RequestMethod.GET)
  @Override
  public String deviceProfileAsYamlForName(@PathVariable String name) {
    try {
      DeviceProfile profile = dao.getByName(name);
      if (profile != null)
        return new Yaml().dumpAs(profile, Tag.MAP, FlowStyle.AUTO);
      else
        throw new NotFoundException(DeviceProfile.class.toString(), name);
    } catch (NotFoundException nfE) {
      throw nfE;
    } catch (Exception e) {
      logger.error("Error getting device profiles:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return all profiles sorted by id. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns LimitExceededException (HTTP 413) if the number returned exceeds
   * the max limit.
   * 
   * @return list of profiles
   */
  @RequestMapping(method = RequestMethod.GET)
  @Override
  public List<DeviceProfile> deviceProfiles() {
    try {
      if (repos.count() > maxLimit) {
        logger.error("Max limit exceeded with request for profils");
        throw new LimitExceededException("DeviceProfile");
      }
      Sort sort = new Sort(Sort.Direction.DESC, "_id");
      return repos.findAll(sort);
    } catch (LimitExceededException lE) {
      throw lE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Find all DeviceProfiles with a manufacture attribute matching that provided. List may be empty
   * if no profiles match. Returns ServiceException (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param manufacturer - manufacturer to match
   * @return List of DeviceProfile matching on specified manufacturer.
   */
  @RequestMapping(method = RequestMethod.GET, value = "/manufacturer/{manufacturer:.+}")
  @Override
  public List<DeviceProfile> deviceProfilesByManufacturer(@PathVariable String manufacturer) {
    try {
      return repos.findByManufacturer(manufacturer);
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Find all DeviceProfiles with a model attribute matching that provided. List may be empty if no
   * profiles match. Returns ServiceException (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param model - model to match
   * @return List of DeviceProfile matching on specified model.
   */
  @RequestMapping(method = RequestMethod.GET, value = "/model/{model:.+}")
  @Override
  public List<DeviceProfile> deviceProfilesByModel(@PathVariable String model) {
    try {
      return repos.findByModel(model);
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Find all DeviceProfiles with a manufacture or model attribute matching that provided (either
   * matching provides a hit). List may be empty if no profiles match. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param manufacturer - manufacturer to match
   * @param model - model to match
   * 
   * @return List of DeviceProfile matching on specified manufacturer and/or model.
   */
  @RequestMapping(method = RequestMethod.GET,
      value = "/manufacturer/{manufacturer:.+}/model/{model:.+}")
  @Override
  public List<DeviceProfile> deviceProfilesByManufacturerOrModel(@PathVariable String manufacturer,
      @PathVariable String model) {
    try {
      return repos.findByManufacturerOrModel(manufacturer, model);
    } catch (Exception e) {
      throw new ServiceException(e);
    }
  }

  /**
   * Find all DeviceProfiles having at least one label matching the label provided. List may be
   * empty if no profiles match. Returns ServiceException (HTTP 503) for unknown or unanticipated
   * issues.
   * 
   * @param label - label to be matched
   * 
   * @return List of DeviceProfile matching on specified label
   */
  @RequestMapping(method = RequestMethod.GET, value = "/label/{label:.+}")
  @Override
  public List<DeviceProfile> deviceProfilesByLabel(@PathVariable String label) {
    try {
      return repos.findByLabelsIn(label);
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Add a new DeviceProfile (and associated Command objects) - name must be unique. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * DataValidationException (HTTP 409) if an associated command's name is a duplicate for the
   * profile.
   * 
   * @param DeviceProfile object
   * @return database generated identifier for the new device profile
   */
  @RequestMapping(method = RequestMethod.POST)
  @Override
  public String add(@RequestBody DeviceProfile deviceProfile) {
    if (deviceProfile == null)
      throw new ServiceException(new DataValidationException("No device profile data provided"));
    try {
      dao.checkCommandNames(deviceProfile.getCommands());
      saveAssociatedCommands(deviceProfile);
      repos.save(deviceProfile);
      return deviceProfile.getId();
    } catch (DuplicateKeyException dE) {
      throw new DataValidationException("Name is not unique: " + deviceProfile.getName());
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error adding device profile:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Add a new DeviceProfile (and associated Command objects) via YAML profile file - name must be
   * unique. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * DataValidationException (HTTP 409) if an associated command's name is a duplicate for the
   * profile. Returns ClientException (HTTP 400) if the YAML file is empty.
   * 
   * 
   * @param file - YAML file containing the profile
   * @return database generated identifier for the new device profile
   */
  @RequestMapping(value = "/uploadfile", method = RequestMethod.POST)
  @Override
  public String uploadYamlFile(@RequestParam("file") MultipartFile file) {
    try {
      if (!file.isEmpty()) {
        String yamlContent = new String(file.getBytes());
        Yaml yaml = new Yaml();
        DeviceProfile deviceProfile = yaml.loadAs(yamlContent, DeviceProfile.class);
        dao.checkCommandNames(deviceProfile.getCommands());
        saveAssociatedCommands(deviceProfile);
        repos.save(deviceProfile);
        return deviceProfile.getId();
      } else {
        throw new ClientException("File is empty");
      }
    } catch (ClientException cE) {
      throw cE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error uploading device profile from YAML:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Add a new DeviceProfile (and associated Command objects) via YAML content - name must be
   * unique. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * DataValidationException (HTTP 409) if an associated command's name is a duplicate for the
   * profile.
   * 
   * @param yamlContent - YAML profile content
   * @return database generated identifier for the new device profile
   */
  @RequestMapping(value = "/upload", method = RequestMethod.POST)
  @Override
  public String uploadYaml(@RequestBody String yamlContent) {
    try {
      Yaml yaml = new Yaml();
      DeviceProfile deviceProfile = (DeviceProfile) yaml.load(yamlContent);
      dao.checkCommandNames(deviceProfile.getCommands());
      saveAssociatedCommands(deviceProfile);
      repos.save(deviceProfile);
      return deviceProfile.getId();
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error uploading device profile from YAML:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the DeviceProfile identified by the id or name stored in the object provided. Id is used
   * first, name is used second for identification purposes. Associated commands must be updated
   * directly. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if the profile cannot be found by the identifier provided.
   * 
   * @param profile2 - object holding the identifier and new values for the DeviceProfile
   * @return boolean indicating success of the update
   */
  @RequestMapping(method = RequestMethod.PUT)
  @Override
  public boolean update(@RequestBody DeviceProfile profile2) {
    if (profile2 == null)
      throw new ServiceException(new DataValidationException("No device profile data provided"));
    try {
      DeviceProfile profile = dao.getByIdOrName(profile2);
      if (profile == null) {
        logger
            .error("Request to update with non-existent or unidentified device profile (id/name):  "
                + profile2.getId() + "/" + profile2.getName());
        throw new NotFoundException(Device.class.toString(), profile2.getId());
      }
      updateDeviceProfile(profile2, profile);
      notifyAssociates(profile, Action.PUT);
      return true;
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (ClientException cE) {
      throw cE;
    } catch (Exception e) {
      logger.error("Error updating device profile:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private void updateDeviceProfile(DeviceProfile from, DeviceProfile to) {
    if (from.getDescription() != null)
      to.setDescription(from.getDescription());
    if (from.getLabels() != null)
      to.setLabels(from.getLabels());
    if (from.getManufacturer() != null)
      to.setDescription(from.getDescription());
    if (from.getModel() != null)
      to.setModel(from.getModel());
    if (from.getObjects() != null)
      to.setObjects(from.getObjects());
    if (from.getOrigin() != 0)
      to.setOrigin(from.getOrigin());
    if (from.getName() != null)
      to.setName(from.getName());
    if (from.getDeviceResources() != null)
      to.setDeviceResources(from.getDeviceResources());
    if (from.getResources() != null)
      to.setResources(from.getResources());
    if (from.getCommands() != null) {
      dao.checkCommandNames(from.getCommands());
      // taking lazy approach to commands - remove them all and add them
      // all back in. TODO - someday make this a two phase commit so
      // commands don't get wiped out before profile
      deleteAssociatedCommands(to);
      saveAssociatedCommands(from);
      to.setCommands(from.getCommands());
      repos.save(to);
    }
  }

  /**
   * Remove the DeviceProfile designated by database generated id. This does not remove associated
   * commands. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if the device profile cannot be found by the identifier provided.
   * Returns DataValidationException (HTTP 413) if devices still reference the profile.
   * 
   * @param database generated id for the device profile
   * @return boolean indicating success of the remove operation
   */
  @RequestMapping(value = "/id/{id}", method = RequestMethod.DELETE)
  @Override
  public boolean delete(@PathVariable String id) {
    try {
      DeviceProfile profile = repos.findOne(id);
      if (profile == null) {
        logger.error("Request to delete with non-existent profile by id:  " + id);
        throw new NotFoundException(DeviceProfile.class.toString(), id);
      }
      return deleteDeviceProfile(profile);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error removing profile:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Remove the DeviceProfile designated by unique name. This does not remove associated commands.
   * Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if the device profile cannot be found by the name provided.
   * Returns DataValidationException (HTTP 413) if devices still reference the profile.
   * 
   * @param unique name of the device profile
   * @return boolean indicating success of the remove operation
   */
  @RequestMapping(value = "/name/{name:.+}", method = RequestMethod.DELETE)
  @Override
  public boolean deleteByName(@PathVariable String name) {
    try {
      DeviceProfile profile = dao.getByName(name);
      if (profile == null) {
        logger.error("Request to delete with unknown profile by name:  " + name);
        throw new NotFoundException(DeviceProfile.class.toString(), name);
      }
      return deleteDeviceProfile(profile);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error removing profile:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private boolean deleteDeviceProfile(DeviceProfile profile) {
    if (associatedDevices(profile))
      throw new DataValidationException(
          "Cannot delete profile.  Associated devices still associated to profile with id: "
              + profile.getId());
    if (associatedProvisionWatchers(profile))
      throw new DataValidationException(
          "Cannot delete profile.  Associated provision watchers still associated to profile with id: "
              + profile.getId());
    deleteAssociatedCommands(profile);
    repos.delete(profile);
    notifyAssociates(profile, Action.DELETE);
    return true;
  }

  private boolean associatedDevices(DeviceProfile profile) {
    return !(deviceRepos.findByProfile(profile).isEmpty());
  }

  private boolean associatedProvisionWatchers(DeviceProfile profile) {
    return !(watcherRepos.findByProfile(profile).isEmpty());
  }

  private void saveAssociatedCommands(DeviceProfile profile) {
    List<Command> commands = profile.getCommands();
    if (commands != null && !commands.isEmpty()) {
      Set<String> commandNames =
          commands.stream().map(Command::getName).collect(Collectors.toSet());
      if (commandNames.size() < profile.getCommands().size())
        throw new DataValidationException("Profile command names not unique");
      profile.getCommands().stream().forEach(c -> commandRepos.save(c));
    }
  }

  private void deleteAssociatedCommands(DeviceProfile profile) {
    List<Command> commands = profile.getCommands();
    if (commands != null && !commands.isEmpty()) {
      profile.getCommands().stream().forEach(c -> commandRepos.delete(c));
    }
  }

  private void notifyAssociates(DeviceProfile profile, Action action) {
    callback.callback(dao.getOwningServices(profile), profile.getId(), action, ActionType.PROFILE);
  }

}

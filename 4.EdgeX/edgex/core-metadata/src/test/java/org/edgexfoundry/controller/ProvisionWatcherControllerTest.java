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

package org.edgexfoundry.controller;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.when;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

import org.edgexfoundry.controller.impl.CallbackExecutor;
import org.edgexfoundry.controller.impl.ProvisionWatcherControllerImpl;
import org.edgexfoundry.dao.DeviceProfileDao;
import org.edgexfoundry.dao.DeviceServiceDao;
import org.edgexfoundry.dao.ProvisionWatcherRepository;
import org.edgexfoundry.domain.meta.DeviceProfile;
import org.edgexfoundry.domain.meta.DeviceService;
import org.edgexfoundry.domain.meta.ProvisionWatcher;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.edgexfoundry.test.category.RequiresNone;
import org.edgexfoundry.test.data.ProfileData;
import org.edgexfoundry.test.data.ProvisionWatcherData;
import org.edgexfoundry.test.data.ServiceData;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.data.domain.Sort;

@Category(RequiresNone.class)
public class ProvisionWatcherControllerTest {


  private static final String LIMIT_PROPERTY = "maxLimit";
  private static final int MAX_LIMIT = 100;

  private static final String TEST_ID = "123";
  private static final String TEST_ERR_MSG = "test message";

  @InjectMocks
  private ProvisionWatcherControllerImpl controller;

  @Mock
  ProvisionWatcherRepository repos;

  @Mock
  private DeviceProfileDao profileDao;

  @Mock
  private DeviceServiceDao serviceDao;

  @Mock
  private CallbackExecutor callback;

  private ProvisionWatcher watcher;

  @Before
  public void setup() throws Exception {
    MockitoAnnotations.initMocks(this);
    setControllerMAXLIMIT(MAX_LIMIT);
    watcher = ProvisionWatcherData.newTestInstance();
    watcher.setId(TEST_ID);
  }

  @Test
  public void testWatcher() {
    when(repos.findOne(TEST_ID)).thenReturn(watcher);
    assertEquals("Provision Watcher returned is not as expected", watcher,
        controller.watcher(TEST_ID));
  }

  @Test(expected = NotFoundException.class)
  public void testWatcherNotFound() {
    controller.watcher(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testWatcherException() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.watcher(TEST_ID);
  }

  @Test
  public void testWatchers() {
    List<ProvisionWatcher> wats = new ArrayList<>();
    wats.add(watcher);
    when(repos.findAll(any(Sort.class))).thenReturn(wats);
    when(repos.count()).thenReturn(1L);
    List<ProvisionWatcher> watchers = controller.watchers();
    assertEquals("Number of watchers returned does not matched expected number", 1,
        watchers.size());
    assertEquals("Watchers returned is not as expected", watcher, watchers.get(0));
  }

  @Test(expected = LimitExceededException.class)
  public void testDeviceServicesMaxLimit() {
    List<ProvisionWatcher> wats = new ArrayList<>();
    wats.add(watcher);
    when(repos.count()).thenReturn(1000L);
    controller.watchers();
  }

  @Test(expected = ServiceException.class)
  public void testDeviceServicesException() {
    List<ProvisionWatcher> wats = new ArrayList<>();
    wats.add(watcher);
    when(repos.findAll(any(Sort.class))).thenThrow(new RuntimeException(TEST_ERR_MSG));
    when(repos.count()).thenReturn(1L);
    controller.watchers();
  }

  @Test
  public void testWatherForName() {
    when(repos.findByName(ProvisionWatcherData.NAME)).thenReturn(watcher);
    assertEquals("Watcher returned is not as expected", watcher,
        controller.watcherForName(ProvisionWatcherData.NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testWatchersForNameNotFound() {
    controller.watcherForName(ProvisionWatcherData.NAME);
  }

  @Test(expected = ServiceException.class)
  public void testWatchersForNameException() {
    when(repos.findByName(ProvisionWatcherData.NAME)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.watcherForName(ProvisionWatcherData.NAME);
  }

  @Test
  public void testWatchersForProfile() {
    DeviceProfile profile = new DeviceProfile();
    List<ProvisionWatcher> wats = new ArrayList<>();
    wats.add(watcher);
    when(profileDao.getById(TEST_ID)).thenReturn(profile);
    when(repos.findByProfile(profile)).thenReturn(wats);
    List<ProvisionWatcher> watchers = controller.watchersForProfile(TEST_ID);
    assertEquals("Number of watchers returned does not matched expected number", 1,
        watchers.size());
    assertEquals("Watcher returned is not as expected", watcher, watchers.get(0));
  }

  @Test(expected = NotFoundException.class)
  public void testWatcherForProfileNoneFound() {
    when(profileDao.getById(TEST_ID)).thenReturn(null);
    controller.watchersForProfile(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceForProfileException() {
    when(profileDao.getById(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.watchersForProfile(TEST_ID);
  }

  @Test
  public void testWatcherForProfileByName() {
    DeviceProfile profile = new DeviceProfile();
    List<ProvisionWatcher> wats = new ArrayList<>();
    wats.add(watcher);
    when(profileDao.getByName(ProfileData.TEST_PROFILE_NAME)).thenReturn(profile);
    when(repos.findByProfile(profile)).thenReturn(wats);
    List<ProvisionWatcher> watchers =
        controller.watchersForProfileByName(ProfileData.TEST_PROFILE_NAME);
    assertEquals("Number of watchers returned does not matched expected number", 1,
        watchers.size());
    assertEquals("Watchers returned is not as expected", watcher, watchers.get(0));
  }

  @Test(expected = NotFoundException.class)
  public void testWatcherForProfileByNameNoneFound() {
    when(profileDao.getByName(ProfileData.TEST_PROFILE_NAME)).thenReturn(null);
    controller.watchersForProfileByName(ProfileData.TEST_PROFILE_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testWatcherForProfileByNameException() {
    when(profileDao.getByName(ProfileData.TEST_PROFILE_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.watchersForProfileByName(ProfileData.TEST_PROFILE_NAME);
  }

  @Test
  public void testWatchersByService() {
    DeviceService service = new DeviceService();
    List<ProvisionWatcher> wats = new ArrayList<>();
    wats.add(watcher);
    when(serviceDao.getById(TEST_ID)).thenReturn(service);
    when(repos.findByService(service)).thenReturn(wats);
    List<ProvisionWatcher> watchers = controller.watcherForService(TEST_ID);
    assertEquals("Number of watchers returned does not matched expected number", 1,
        watchers.size());
    assertEquals("Watcher returned is not as expected", watcher, watchers.get(0));
  }

  @Test(expected = NotFoundException.class)
  public void testWatchersByServiceNoneFound() {
    when(serviceDao.getById(TEST_ID)).thenReturn(null);
    controller.watcherForService(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testWatchersByServiceException() {
    when(serviceDao.getById(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.watcherForService(TEST_ID);
  }

  @Test
  public void testWatchersByServiceName() {
    DeviceService service = new DeviceService();
    List<ProvisionWatcher> wats = new ArrayList<>();
    wats.add(watcher);
    when(serviceDao.getByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(service);
    when(repos.findByService(service)).thenReturn(wats);
    List<ProvisionWatcher> watchers =
        controller.watcherForServiceByName(ServiceData.TEST_SERVICE_NAME);
    assertEquals("Number of watchers returned does not matched expected number", 1,
        watchers.size());
    assertEquals("Watcher returned is not as expected", watcher, watchers.get(0));
  }

  @Test(expected = NotFoundException.class)
  public void testWatcherByServiceNameNoneFound() {
    when(serviceDao.getByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(null);
    controller.watcherForServiceByName(ServiceData.TEST_SERVICE_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceByServiceNameException() {
    when(serviceDao.getByName(ServiceData.TEST_SERVICE_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.watcherForServiceByName(ServiceData.TEST_SERVICE_NAME);
  }

  @Test
  public void testWatcherByIdentifier() {
    List<ProvisionWatcher> wats = new ArrayList<>();
    wats.add(watcher);
    when(repos.findByIdendifierKeyValue("identifiers." + ProvisionWatcherData.KEY1,
        ProvisionWatcherData.VAL1)).thenReturn(wats);
    List<ProvisionWatcher> watchers =
        controller.watchersForIdentifier(ProvisionWatcherData.KEY1, ProvisionWatcherData.VAL1);
    assertEquals("Number of watchers returned does not matched expected number", 1,
        watchers.size());
    assertEquals("Watchers returned is not as expected", watcher, watchers.get(0));
  }

  @Test(expected = ServiceException.class)
  public void testWatcherByIdentifierException() {
    List<ProvisionWatcher> wats = new ArrayList<>();
    wats.add(watcher);
    when(repos.findByIdendifierKeyValue("identifiers." + ProvisionWatcherData.KEY1,
        ProvisionWatcherData.VAL1)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.watchersForIdentifier(ProvisionWatcherData.KEY1, ProvisionWatcherData.VAL1);
  }

  @Test
  public void testAdd() {
    DeviceService service = ServiceData.newTestInstance();
    DeviceProfile profile = ProfileData.newTestInstance();
    watcher.setService(service);
    watcher.setProfile(profile);
    when(serviceDao.getByIdOrName(service)).thenReturn(service);
    when(profileDao.getByIdOrName(profile)).thenReturn(profile);
    when(repos.save(watcher)).thenReturn(watcher);
    assertEquals("Watcher returned was not the same as added", TEST_ID, controller.add(watcher));
  }

  @Test(expected = ServiceException.class)
  public void testAddWithNull() {
    controller.add(null);
  }

  @Test(expected = DataValidationException.class)
  public void testAddWithDuplicatKey() {
    DeviceService service = ServiceData.newTestInstance();
    DeviceProfile profile = ProfileData.newTestInstance();
    watcher.setService(service);
    watcher.setProfile(profile);
    when(serviceDao.getByIdOrName(service)).thenReturn(service);
    when(profileDao.getByIdOrName(profile)).thenReturn(profile);

    when(repos.save(watcher)).thenThrow(new DuplicateKeyException(TEST_ERR_MSG));
    controller.add(watcher);
  }

  @Test(expected = ServiceException.class)
  public void testAddException() {
    DeviceService service = ServiceData.newTestInstance();
    DeviceProfile profile = ProfileData.newTestInstance();
    watcher.setService(service);
    watcher.setProfile(profile);
    when(serviceDao.getByIdOrName(service)).thenReturn(service);
    when(profileDao.getByIdOrName(profile)).thenReturn(profile);
    when(repos.save(watcher)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.add(watcher);
  }

  @Test(expected = DataValidationException.class)
  public void testAddProfileNotFound() {
    DeviceService service = ServiceData.newTestInstance();
    DeviceProfile profile = ProfileData.newTestInstance();
    watcher.setService(service);
    watcher.setProfile(profile);
    when(serviceDao.getByIdOrName(service)).thenReturn(service);
    when(profileDao.getByIdOrName(profile)).thenReturn(null);
    controller.add(watcher);
  }

  @Test(expected = DataValidationException.class)
  public void testAddServiceNotFound() {
    DeviceService service = ServiceData.newTestInstance();
    DeviceProfile profile = ProfileData.newTestInstance();
    watcher.setService(service);
    watcher.setProfile(profile);
    when(serviceDao.getByIdOrName(service)).thenReturn(null);
    controller.add(watcher);
  }

  @Test
  public void testUpdate() {
    when(repos.findOne(TEST_ID)).thenReturn(watcher);
    when(repos.save(watcher)).thenReturn(watcher);
    assertTrue("Provision watcher was not updated", controller.update(watcher));
  }

  @Test
  public void testUpdateByName() {
    watcher.setId(null);
    when(repos.findByName(ProvisionWatcherData.NAME)).thenReturn(watcher);
    when(repos.save(watcher)).thenReturn(watcher);
    assertTrue("Provision watcher was not updated", controller.update(watcher));
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateWatcherNotFound() {
    when(repos.findOne(TEST_ID)).thenReturn(null);
    controller.update(watcher);
  }

  @Test(expected = ServiceException.class)
  public void testUpdateException() {
    when(repos.findOne(TEST_ID)).thenReturn(watcher);
    when(repos.save(watcher)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.update(watcher);
  }

  @Test(expected = ServiceException.class)
  public void testUpdateWithNull() {
    controller.update(null);
  }

  @Test
  public void testDelete() {
    when(repos.findOne(TEST_ID)).thenReturn(watcher);
    assertTrue("Provision watcher was not deleted", controller.delete(TEST_ID));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteNotFound() {
    when(repos.findOne(TEST_ID)).thenReturn(null);
    controller.delete(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testDeleteException() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    assertTrue("Provision watcher was not deleted", controller.delete(TEST_ID));
  }

  @Test
  public void testDeleteByName() {
    when(repos.findByName(ProvisionWatcherData.NAME)).thenReturn(watcher);
    assertTrue("Provision watcher was not deleted",
        controller.deleteByName(ProvisionWatcherData.NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteByNameNotFound() {
    when(repos.findByName(ProvisionWatcherData.NAME)).thenReturn(null);
    controller.deleteByName(ProvisionWatcherData.NAME);
  }

  @Test(expected = ServiceException.class)
  public void testDeleteByNameException() {
    when(repos.findByName(ProvisionWatcherData.NAME)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    assertTrue("Device service was not deleted",
        controller.deleteByName(ProvisionWatcherData.NAME));
  }

  private void setControllerMAXLIMIT(int newLimit) throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField(LIMIT_PROPERTY);
    temp.setAccessible(true);
    temp.set(controller, newLimit);
  }
}
